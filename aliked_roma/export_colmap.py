"""Export immutable ALIKED keypoints and RoMa row-ID matches to COLMAP 4.x.

python -m fixed_sfm.export_colmap --help
python export_colmap.py --help  (from fixed_sfm/)

The new output directory is published only after validation succeeds. Raw
matches require COLMAP geometric_verifier before mapper; the initial RoMa F
matrix is deliberately not written as a COLMAP TwoViewGeometry.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import fcntl
from hashlib import sha256
import json
import logging
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import tempfile
from time import perf_counter

import numpy as np
from PIL import Image

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = 'fixed_sfm'

from .extract import atomic_json, source_path
from .features import COORDINATES, check_name, digest_json
from .match import FeatureBank
from .match_store import MATCH_SCHEMA, canonical_pair, load_matches, match_path, pair_id

LOG = logging.getLogger('fixed_sfm.export_colmap')
MAX_IMAGE_ID = 2147483647


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--features-dir', type=Path, required=True)
    p.add_argument('--matches-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True, help='NEW snapshot directory; existing directories are never overwritten')
    p.add_argument('--images-dir', type=Path, help='Default: original image directory recorded in the feature bank')
    p.add_argument('--camera-model', help='Shared camera model, e.g. PINHOLE; requires --camera-params')
    p.add_argument('--camera-params', help='Original-resolution intrinsics, e.g. fx,fy,cx,cy for PINHOLE')
    p.add_argument('--camera-config', type=Path, help='JSON cameras/default_camera_id/image_cameras; alternative to shared-camera flags')
    p.add_argument('--match-selection', choices=('inliers', 'all'), default='inliers',
                   help='Initial F inliers (default), or all reciprocal matches; both require native geometry verification')
    p.add_argument('--min-matches', type=int, default=15, help='Minimum selected matches per exported pair (>=8)')
    p.add_argument('--pair-scope', choices=('selected', 'completed'), default='selected',
                   help='selected: require every pair in pairs_used.json; completed: export the cumulative match index')
    p.add_argument('--dry-run', action='store_true', help='Validate all inputs and report counts without writing files')
    args = p.parse_args(argv)
    if args.min_matches < 8:
        p.error('min-matches must be >=8')
    if args.camera_config is not None:
        if args.camera_model is not None or args.camera_params is not None:
            p.error('Use --camera-config OR --camera-model/--camera-params')
    elif not args.camera_model or not args.camera_params:
        p.error('Specify --camera-config or BOTH --camera-model and --camera-params; intrinsics are not guessed')
    for name in ('features_dir', 'matches_dir', 'output_dir', 'images_dir', 'camera_config'):
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).expanduser().resolve())
    for root in (args.features_dir, args.matches_dir):
        if args.output_dir.is_relative_to(root) or root.is_relative_to(args.output_dir):
            p.error('output-dir must be separate from feature and match inputs')
    return args


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def load_inputs(args):
    bank = FeatureBank(args.features_dir)
    if not bank.rows:
        raise ValueError('Feature bank contains no images')
    config = read_json(args.matches_dir/'run_config.json')
    run_id = config['run_id']
    if digest_json(config['config']) != run_id:
        raise ValueError('Match configuration checksum is invalid')
    if config['config'].get('feature_set_id') != bank.set_id:
        raise ValueError('Matching run and feature bank identities differ')
    if config['config'].get('schema_version') != MATCH_SCHEMA:
        raise ValueError('Unsupported matching configuration schema')
    index = read_json(args.matches_dir/'index.json')
    if index.get('schema_version') != MATCH_SCHEMA or index.get('run_id') != run_id or not isinstance(index.get('pairs'), dict):
        raise ValueError('Match index/configuration disagree')
    rows = index['pairs']
    if args.pair_scope == 'selected':
        raw_pairs = read_json(args.matches_dir/'pairs_used.json')
    else:
        raw_pairs = [[row['image0'], row['image1']] for row in rows.values()]
    if not isinstance(raw_pairs, list) or not raw_pairs:
        raise ValueError('No image pairs selected for export')
    pairs = []
    for pair in raw_pairs:
        if not isinstance(pair, list) or len(pair) != 2 or tuple(pair) != canonical_pair(*pair):
            raise ValueError(f'Invalid or noncanonical image pair: {pair!r}')
        a, b = pair
        if a not in bank.rows or b not in bank.rows:
            raise ValueError(f'Match pair refers to an unindexed image: {pair!r}')
        row = rows.get(pair_id(a, b))
        if row is None:
            raise ValueError(f'Selected pair is not completed/indexed: {a} <-> {b}. Finish matching first, or explicitly use --pair-scope completed.')
        if (row['image0'], row['image1']) != (a, b):
            raise ValueError(f'Match index pair-ID mismatch: {a} <-> {b}')
        expected_file = match_path(args.matches_dir, a, b).relative_to(args.matches_dir).as_posix()
        if row['match_file'] != expected_file:
            raise ValueError(f'Match index path mismatch: {a} <-> {b}')
        pairs.append((a, b))
    if len(set(pairs)) != len(pairs):
        raise ValueError('Pair selection contains duplicates')
    return bank, run_id, rows, sorted(pairs)


def load_cameras(args, image_sizes, lib):
    """Explicit calibration only. Dimensions use the stored image orientation."""
    if args.camera_config is None:
        sizes = set(image_sizes.values())
        if len(sizes) != 1:
            raise ValueError('Shared camera requires identical image dimensions; use --camera-config')
        w, h = next(iter(sizes))
        raw = [{'id': 1, 'model': args.camera_model, 'width': w, 'height': h,
                'params': [float(v.strip()) for v in args.camera_params.split(',')]}]
        assignments = {name: 1 for name in image_sizes}
    else:
        config = read_json(args.camera_config)
        raw = config['cameras']
        explicit = config.get('image_cameras', {})
        unknown = set(explicit)-set(image_sizes)
        if unknown:
            raise ValueError(f'Camera config contains unknown images: {sorted(unknown)[:5]}')
        assignments = {}
        for name in image_sizes:
            cid = explicit.get(name, config.get('default_camera_id'))
            if type(cid) is not int:
                raise ValueError(f'Missing or invalid camera assignment for {name}')
            assignments[name] = cid
    if not isinstance(raw, list) or not raw:
        raise ValueError('Camera config needs a nonempty cameras list')
    cameras = {}
    for row in raw:
        cid, w, h = row['id'], row['width'], row['height']
        if any(type(x) is not int or x <= 0 for x in (cid, w, h)) or cid >= MAX_IMAGE_ID:
            raise ValueError('Camera IDs/dimensions must be positive integers')
        if cid in cameras:
            raise ValueError(f'Duplicate camera ID: {cid}')
        params = np.asarray(row['params'], dtype=np.float64)
        if params.ndim != 1 or not len(params) or not np.isfinite(params).all():
            raise ValueError(f'Invalid camera parameters for ID {cid}')
        camera = lib.Camera.create_from_model_name(cid, str(row['model']).upper(), 1., w, h)
        if not camera.set_params_from_string(','.join(format(v, '.17g') for v in params)) or not camera.verify_params():
            raise ValueError(f'Camera {cid}: parameter count does not match {row["model"]}')
        if (camera.params[camera.focal_length_idxs()] <= 0).any():
            raise ValueError(f'Camera {cid}: focal lengths must be positive')
        camera.has_prior_focal_length = True
        if hasattr(lib, 'CameraSource'):
            camera.source = lib.CameraSource.USER
        cameras[cid] = camera
    for name, size in image_sizes.items():
        cid = assignments[name]
        if cid not in cameras:
            raise ValueError(f'{name} refers to unknown camera ID {cid}')
        c = cameras[cid]
        if size != (c.width, c.height):
            raise ValueError(f'Image size / camera dimensions disagree: {name}, image={size}, camera={(c.width, c.height)}')
    unused = set(cameras)-set(assignments.values())
    if unused:
        raise ValueError(f'Camera config contains unused camera IDs: {sorted(unused)}')
    return cameras, assignments


def verify_source(root, feature):
    path = source_path(root, feature.image_name)
    h = sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    if h.hexdigest() != feature.source_sha256:
        raise ValueError(f'Source image content changed since extraction: {feature.image_name}')
    with Image.open(path) as im:
        if im.size != tuple(feature.image_size):
            raise ValueError(f'Source image dimensions disagree with bank: {feature.image_name}')


def add_cameras(db, lib, cameras):
    sensors = {}
    for cid, camera in sorted(cameras.items()):
        if db.write_camera(camera, use_camera_id=True) != cid:
            raise RuntimeError('COLMAP changed a requested camera ID')
        sensor = lib.sensor_t(lib.SensorType.CAMERA, cid)
        rig = lib.Rig()
        rig.rig_id = cid
        rig.add_ref_sensor(sensor)
        db.write_rig(rig, use_rig_id=True)
        sensors[cid] = sensor
    return sensors


def write_image(db, lib, name, row, feature, sensors):
    iid, cid = row['image_id'], row['camera_id']
    frame = lib.Frame()
    frame.frame_id, frame.rig_id = iid, cid
    frame.add_data_id(lib.data_t(sensors[cid], iid))
    db.write_frame(frame, use_frame_id=True)
    image = lib.Image(name=name, camera_id=cid, image_id=iid)
    image.frame_id = iid
    if db.write_image(image, use_image_id=True) != iid:
        raise RuntimeError(f'COLMAP changed image ID for {name}')
    # Already original-resolution COLMAP coordinates. No +0.5, sorting,
    # merging, or filtering: the original bank row remains the feature ID.
    db.write_keypoints(iid, np.ascontiguousarray(feature.keypoints))
    # Descriptors are not needed by geometric_verifier or mapper. Do not
    # label ALIKED descriptors as SIFT or create fabricated descriptors.


def checked_pair(root, a, b, run_id, set_id, refs, indexed):
    pair = load_matches(root, a, b)  # validates NPZ checksum, shapes, IDs
    expected = dict(images=[a, b], feature_set_id=set_id, run_id=run_id,
                    feature_ids=[refs[a]['feature_id'], refs[b]['feature_id']],
                    keypoint_counts=[refs[a]['keypoints'], refs[b]['keypoints']])
    for key, value in expected.items():
        if pair.metadata.get(key) != value:
            raise ValueError(f'Pair {key} differs from verified fixed bank/config: {a} <-> {b}')
    row = indexed[pair_id(a, b)]
    if (row['content_id'] != pair.content_id or row['matches'] != len(pair.matches)
            or row['inliers'] != int(pair.inliers.sum())
            or [row['keypoints0'], row['keypoints1']] != expected['keypoint_counts']):
        raise ValueError(f'Match index content mismatch: {a} <-> {b}')
    return pair


def audit_database(path, expected):
    """Read-only schema/integrity audit after native COLMAP closes the DB."""
    with sqlite3.connect(path.as_uri()+'?mode=ro') as con:
        if con.execute('PRAGMA integrity_check').fetchone() != ('ok',):
            raise RuntimeError('COLMAP database failed SQLite integrity_check')
        if con.execute('PRAGMA foreign_key_check').fetchone() is not None:
            raise RuntimeError('COLMAP database has invalid foreign keys')
        result = {}
        for table in ('cameras', 'rigs', 'frames', 'images', 'keypoints', 'descriptors', 'matches', 'two_view_geometries'):
            result[table] = con.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]
        result['total_keypoints'] = con.execute('SELECT COALESCE(SUM(rows),0) FROM keypoints').fetchone()[0]
        result['total_matches'] = con.execute('SELECT COALESCE(SUM(rows),0) FROM matches').fetchone()[0]
    if result != expected:
        raise RuntimeError(f'COLMAP database counts disagree: {result!r} != {expected!r}')
    return result


@contextmanager
def export_lock(parent):
    with (parent/'.fixed_sfm_export.lock').open('a') as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'Another COLMAP export is writing in {parent}') from exc
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def build_snapshot(args, lib, bank, run_id, indexed, pairs, destination):
    started = perf_counter()
    names = sorted(bank.rows)
    if len(names) >= MAX_IMAGE_ID:
        raise ValueError('Too many images for COLMAP pair IDs')
    # Read/verify every bank, including images with zero keypoints. Keep only
    # metadata in memory; coordinates are streamed again for the DB writer.
    refs = {}
    for i, name in enumerate(names, 1):
        check_name(name)
        feature = bank.get(name)
        verify_source(args.images_dir, feature)
        refs[name] = dict(image_id=i, feature_id=feature.feature_id,
                          keypoints=len(feature.keypoints), image_size=feature.image_size.tolist(),
                          source_sha256=feature.source_sha256)
        if i % 200 == 0 or i == len(names):
            LOG.info('Validated fixed banks/source images: %d/%d', i, len(names))
    cameras, assignments = load_cameras(args, {n: tuple(r['image_size']) for n, r in refs.items()}, lib)
    for name in names:
        refs[name]['camera_id'] = assignments[name]
    db = None
    rows, neighbors = [], {name: set() for name in names}
    selected_total = exported_total = exported_pairs = raw_total = inlier_total = 0
    try:
        if destination is not None:
            db = lib.Database.open(destination/'database.db')
            with lib.DatabaseTransaction(db):
                sensors = add_cameras(db, lib, cameras)
                for name in names:
                    feature = bank.get(name)
                    if feature.feature_id != refs[name]['feature_id']:
                        raise ValueError(f'Feature bank changed during export: {name}')
                    write_image(db, lib, name, refs[name], feature, sensors)
        def process_pairs():
            nonlocal selected_total, exported_total, exported_pairs, raw_total, inlier_total
            for i, (a, b) in enumerate(pairs, 1):
                pair = checked_pair(args.matches_dir, a, b, run_id, bank.set_id, refs, indexed)
                matches = pair.matches[pair.inliers] if args.match_selection == 'inliers' else pair.matches
                n = len(matches)
                exported = n >= args.min_matches
                raw_total += len(pair.matches)
                inlier_total += int(pair.inliers.sum())
                selected_total += n
                if exported:
                    if db is not None:
                        db.write_matches(refs[a]['image_id'], refs[b]['image_id'], np.ascontiguousarray(matches))
                    exported_pairs += 1
                    exported_total += n
                    neighbors[a].add(b)
                    neighbors[b].add(a)
                rows.append(dict(image0=a, image1=b, raw_matches=len(pair.matches),
                                 initial_f_inliers=int(pair.inliers.sum()), selected_matches=n,
                                 exported_matches=n if exported else 0,
                                 status='exported' if exported else 'below_min_matches',
                                 content_id=pair.content_id))
                if i % 1000 == 0 or i == len(pairs):
                    LOG.info('Validated pairs: %d/%d; exported %d pairs, %d matches', i, len(pairs), exported_pairs, exported_total)
        if db is None:
            process_pairs()
        else:
            with lib.DatabaseTransaction(db):
                process_pairs()
    finally:
        if db is not None:
            db.close()
    if not exported_pairs:
        raise ValueError('No pair meets --min-matches; no database published')
    counts = dict(cameras=len(cameras), rigs=len(cameras), frames=len(names), images=len(names),
                  keypoints=len(names), descriptors=0, matches=exported_pairs, two_view_geometries=0,
                  total_keypoints=sum(r['keypoints'] for r in refs.values()), total_matches=exported_total)
    camera_report = [dict(id=cid, model=c.model_name, width=c.width, height=c.height,
                          params=c.params.tolist(), has_prior_focal_length=True) for cid, c in sorted(cameras.items())]
    report = dict(schema_version=1, status='validated' if destination is None else 'exported_raw_matches',
                  feature_set_id=bank.set_id, matching_run_id=run_id, coordinates=COORDINATES,
                  features_dir=str(args.features_dir), matches_dir=str(args.matches_dir), images_dir=str(args.images_dir),
                  database_path=str(args.output_dir/'database.db'), pycolmap_version=lib.__version__,
                  pair_scope=args.pair_scope, match_selection=args.match_selection, min_matches=args.min_matches,
                  selected_pairs=len(pairs), exported_pairs=exported_pairs, skipped_pairs=len(pairs)-exported_pairs,
                  input_raw_matches=raw_total, input_initial_f_inliers=inlier_total,
                  selected_matches=selected_total, exported_matches=exported_total,
                  images_without_exported_pairs=[n for n in names if not neighbors[n]],
                  cameras=camera_report, database_counts=counts, requires_geometric_verification=True,
                  elapsed_seconds=perf_counter()-started)
    if destination is not None:
        audit_database(destination/'database.db', counts)
        atomic_json(destination/'image_ids.json', dict(feature_set_id=bank.set_id, images=refs))
        with (destination/'pair_statistics.csv').open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        # Geometric verifier in COLMAP 4.x enumerates the database directly;
        # this list is also useful with older matches_importer workflows.
        with (destination/'pairs.txt').open('w', encoding='utf-8') as f:
            for row in rows:
                if row['status'] == 'exported':
                    f.write(f'{row["image0"]} {row["image1"]}\n')
        atomic_json(destination/'export_report.json', report)
    return report


def run(args):
    # Fail before touching outputs; existing DBs may already contain geometry
    # or a long-running mapper's state. Choose another output directory.
    if args.output_dir.exists():
        raise FileExistsError(f'Output directory already exists: {args.output_dir}. Choose a new --output-dir.')
    bank, run_id, indexed, pairs = load_inputs(args)
    args.images_dir = args.images_dir or Path(bank.config['config']['images_dir']).expanduser().resolve()
    if not args.images_dir.is_dir():
        raise FileNotFoundError(args.images_dir)
    if args.output_dir.is_relative_to(args.images_dir):
        raise ValueError('output-dir must be outside images-dir')
    try:
        import pycolmap
    except ImportError as exc:
        raise RuntimeError('pycolmap is required. Use /home/nvidia/venvs/2dgs_env/bin/python (existing COLMAP 4.x build).') from exc
    if args.dry_run:
        report = build_snapshot(args, pycolmap, bank, run_id, indexed, pairs, None)
    else:
        parent = args.output_dir.parent
        parent.mkdir(parents=True, exist_ok=True)
        with export_lock(parent):
            if args.output_dir.exists():
                raise FileExistsError(args.output_dir)
            temporary = Path(tempfile.mkdtemp(prefix='.colmap_export_', dir=parent))
            try:
                report = build_snapshot(args, pycolmap, bank, run_id, indexed, pairs, temporary)
                if args.output_dir.exists():
                    raise FileExistsError(args.output_dir)
                os.rename(temporary, args.output_dir)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    try:
        run(parse_args())
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        LOG.error('%s', exc)
        raise SystemExit(1) from exc


if __name__ == '__main__':
    main()
