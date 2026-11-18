"""Match fixed ALIKED keypoints with RoMa, preserving original row IDs.

Project root: python -m aliked_roma.match --help
Script directory: python match.py --help
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import csv
from hashlib import sha256
import json
import logging
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = 'aliked_roma'

from .features import COORDINATES, SCHEMA_VERSION, digest_json, load_features
from .extract import atomic_json, output_lock, source_path
from .matching import associate, verify_fundamental
from .match_store import MATCH_SCHEMA, PairMatches, canonical_pair, load_matches, match_path, pair_id, save_matches

LOG = logging.getLogger('aliked_roma.match')


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--features-dir', type=Path, required=True, help='Immutable ALIKED bank from extract.py')
    p.add_argument('--images-dir', type=Path, help='Default: image root recorded in the feature bank')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--pairs-file', type=Path, help='Two image names per line; otherwise sequential pairs from the feature index')
    pairing = p.add_mutually_exclusive_group()
    pairing.add_argument('--pairing-mode', choices=('linear', 'exponential', 'hybrid'), default='linear',
                         help='Generated pair offsets: 1..K, powers 2^0..2^(K-1), or their union')
    pairing.add_argument('--quadratic-overlap', action='store_true',
                         help='COLMAP-style alias for --pairing-mode exponential')
    p.add_argument('--overlap', type=int, default=5,
                   help='K: linear window size, exponential level count, or both in hybrid mode')
    p.add_argument('--max-pairs', type=int, default=0, help='First N canonical sorted pairs; 0 = all')
    p.add_argument('--weights-dir', type=Path, default=Path.home()/'models/RoMa')
    local_roma = Path.home()/'src/RoMa'
    p.add_argument('--roma-repo-dir', type=Path, default=local_roma if local_roma.is_dir() else None)
    p.add_argument('--device', default='cuda')
    p.add_argument('--coarse-res', type=int, default=448)
    p.add_argument('--upsample-res', type=int, default=672)
    p.add_argument('--num-threads', type=int, default=4)
    p.add_argument('--max-assignment-distance', type=float, default=3., help='Anchor assignment radius in ORIGINAL target-image pixels')
    p.add_argument('--nn-ratio', type=float, default=.8, help='Nearest / second-nearest spatial distance; 1 still rejects ties')
    p.add_argument('--min-confidence', type=float, default=.2)
    p.add_argument('--max-cycle-error', type=float, default=3., help='Unsnapped round-trip error in ORIGINAL source-image pixels')
    p.add_argument('--ransac-threshold', type=float, default=2.)
    p.add_argument('--ransac-confidence', type=float, default=.999)
    p.add_argument('--ransac-max-iters', type=int, default=10000)
    p.add_argument('--min-pair-inliers', type=int, default=15, help='Classify weaker pairs as insufficient_inliers; still record them')
    p.add_argument('--dry-run', action='store_true', help='Validate feature banks and list pairs; no torch/model load or output writes')
    args = p.parse_args(argv)
    if args.quadratic_overlap:
        args.pairing_mode = 'exponential'
    if args.overlap < 1 or args.max_pairs < 0 or args.num_threads < 1:
        p.error('overlap and num-threads must be positive; max-pairs must be >=0')
    if args.coarse_res <= 0 or args.coarse_res % 14 or args.upsample_res < 32:
        p.error('coarse-res must be a positive multiple of 14; upsample-res must be >=32')
    values = [args.max_assignment_distance, args.nn_ratio, args.min_confidence, args.max_cycle_error,
              args.ransac_threshold, args.ransac_confidence]
    if not np.isfinite(values).all() or min(args.max_assignment_distance, args.max_cycle_error, args.ransac_threshold) <= 0:
        p.error('distance/error thresholds must be finite and positive')
    if not 0 < args.nn_ratio <= 1 or not 0 <= args.min_confidence <= 1 or not 0 < args.ransac_confidence < 1:
        p.error('invalid ratio, confidence, or RANSAC confidence')
    if args.ransac_max_iters < 1 or args.min_pair_inliers < 8:
        p.error('ransac-max-iters must be positive; min-pair-inliers must be >=8')
    for name in ['features_dir', 'images_dir', 'output_dir', 'pairs_file', 'weights_dir', 'roma_repo_dir']:
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.output_dir.is_relative_to(args.features_dir):
        p.error('output-dir must be separate from the immutable feature bank')
    return args


class FeatureBank:
    """Validate exact bank identities; bound the decompressed feature cache."""
    def __init__(self, root):
        self.root, self.cache = root, OrderedDict()
        config_path, index_path = root/'run_config.json', root/'index.json'
        if not config_path.is_file() or not index_path.is_file():
            raise FileNotFoundError(f'Feature bank needs run_config.json and index.json: {root}. Run extract.py first.')
        self.config = json.loads(config_path.read_text())
        index = json.loads(index_path.read_text())
        self.set_id = self.config['feature_set_id']
        if digest_json(self.config['config']) != self.set_id:
            raise ValueError('Feature bank configuration checksum is invalid')
        if (index.get('schema_version') != SCHEMA_VERSION or index.get('feature_set_id') != self.set_id
                or self.config['config'].get('coordinates') != COORDINATES or not isinstance(index.get('images'), dict)):
            raise ValueError('Feature index/configuration disagree')
        self.rows = index['images']

    def get(self, name):
        if name not in self.rows:
            raise KeyError(f'No indexed fixed keypoints for {name!r}; extract this image first')
        if name not in self.cache:
            bank = load_features(self.root, name, feature_set_id=self.set_id)
            row = self.rows[name]
            if bank.feature_id != row['feature_id'] or len(bank.keypoints) != row['keypoints']:
                raise ValueError(f'Feature index content mismatch for {name}')
            self.cache[name] = bank
            if len(self.cache) > 16:
                self.cache.popitem(last=False)
        self.cache.move_to_end(name)
        return self.cache[name]


def sequential_offsets(num_images, overlap=5, pairing_mode='linear'):
    """COLMAP-style index gaps, clipped before computing large powers.

    overlap=5 yields [1,2,3,4,5] in linear mode and [1,2,4,8,16]
    in exponential mode. Hybrid combines both. This is index distance in
    filename order, not elapsed time or a maximum multi-view track length.
    """
    if overlap < 1:
        raise ValueError('overlap must be positive')
    if pairing_mode not in ('linear', 'exponential', 'hybrid'):
        raise ValueError(f'Unknown pairing mode: {pairing_mode}')
    limit = max(0, num_images-1)
    offsets = set()
    if pairing_mode in ('linear', 'hybrid'):
        offsets.update(range(1, min(overlap, limit)+1))
    if pairing_mode in ('exponential', 'hybrid'):
        offsets.update(1 << level for level in range(min(overlap, limit.bit_length())))
    return sorted(offsets)


def collect_pairs(names, pairs_file=None, overlap=5, max_pairs=0, pairing_mode='linear'):
    names = sorted(names)
    available = set(names)
    if pairs_file is None:
        offsets = sequential_offsets(len(names), overlap, pairing_mode)
        pairs = [(a, names[i+gap]) for i, a in enumerate(names) for gap in offsets if i+gap < len(names)]
    else:
        pairs = []
        for number, line in enumerate(pairs_file.read_text(encoding='utf-8-sig').splitlines(), 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f'{pairs_file}:{number}: expected two image names')
            pairs.append(canonical_pair(*parts))
    pairs = sorted(set(canonical_pair(a, b) for a, b in pairs))
    for pair in pairs:
        for name in pair:
            if name not in available:
                raise ValueError(f'Pair requests {name!r}, which has no indexed fixed keypoints')
    if max_pairs:
        pairs = pairs[:max_pairs]
    if not pairs:
        raise ValueError('No image pairs selected; at least two images with extracted keypoints are needed')
    return pairs


def verified_image_bytes(root, bank):
    data = source_path(root, bank.image_name).read_bytes()
    if sha256(data).hexdigest() != bank.source_sha256:
        raise ValueError(f'Source image changed since ALIKED extraction: {bank.image_name}. Build a new feature bank first.')
    return data


def prepare_output(root, config):
    run_id = digest_json(config)
    expected = dict(run_id=run_id, config=config)
    path = root/'run_config.json'
    if path.exists():
        if json.loads(path.read_text()) != expected:
            raise ValueError('Matching configuration changed. Use a NEW output directory; existing pair results cannot be overwritten.')
    else:
        if any(root.glob('matches/*.npz')) or (root/'index.json').exists():
            raise ValueError('Match files/index exist without run_config.json; use a new output directory')
        atomic_json(path, expected)
    path = root/'index.json'
    index = json.loads(path.read_text()) if path.exists() else dict(schema_version=MATCH_SCHEMA, run_id=run_id, pairs={})
    if index.get('schema_version') != MATCH_SCHEMA or index.get('run_id') != run_id or not isinstance(index.get('pairs'), dict):
        raise ValueError('Pair index/configuration mismatch')
    return run_id, index


def process_pair(args, backend, banks, names, run_id, verifier):
    f0, f1 = (banks.get(n) for n in names)
    # Hash exactly the bytes fed to RoMa, including on resume; no stale-image reuse.
    data0, data1 = verified_image_bytes(args.images_dir, f0), verified_image_bytes(args.images_dir, f1)
    if match_path(args.output_dir, *names).exists():
        result = load_matches(args.output_dir, *names)
        result.bind(f0, f1, run_id)
        return result, 'cached'
    n0, n1 = len(f0.keypoints), len(f1.keypoints)
    if n0 and n1:
        predictions = backend.predict(data0, data1, f0, f1)
        matches, scores, assignment, cycles, stats = associate(f0, f1, predictions,
            max_distance=args.max_assignment_distance, nn_ratio=args.nn_ratio,
            min_confidence=args.min_confidence, max_cycle_error=args.max_cycle_error)
    else:
        matches, scores = np.empty((0, 2), np.uint32), np.empty(0, np.float32)
        assignment, cycles = np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)
        stats = dict(forward_candidates=0, backward_candidates=0, reciprocal=0)
    matrix, inliers = verifier(f0.keypoints[matches[:, 0]], f1.keypoints[matches[:, 1]],
        threshold=args.ransac_threshold, confidence=args.ransac_confidence, max_iters=args.ransac_max_iters,
        seed=int(pair_id(*names)[:8], 16) & 0x7fffffff)
    metadata = dict(images=list(names), feature_set_id=banks.set_id, feature_ids=[f0.feature_id, f1.feature_id],
                    keypoint_counts=[n0, n1], run_id=run_id, stats=stats)
    result = PairMatches(metadata, matches, scores, assignment, cycles, inliers, matrix)
    result.bind(f0, f1, run_id)
    save_matches(args.output_dir, result)
    return result, 'matched'


def association_report(root, pairs):
    """Measure actual reuse on this run's selected, F-inlier match graph."""
    degrees, edges = {}, 0
    for names in pairs:
        r = load_matches(root, *names)
        m = r.matches[r.inliers]
        edges += len(m)
        for side, name in enumerate(names):
            degrees.setdefault(name, np.zeros(r.metadata['keypoint_counts'][side], np.uint32))
            degrees[name][m[:, side]] += 1  # each column is one-to-one
    histogram = np.zeros(1, np.int64)
    for d in degrees.values():
        h = np.bincount(d.astype(np.int64))
        if len(h) > len(histogram):
            histogram = np.pad(histogram, (0, len(h)-len(histogram)))
        histogram[:len(h)] += h
    isolated = 0
    for names in pairs:
        r = load_matches(root, *names)
        m = r.matches[r.inliers]
        isolated += int(((degrees[names[0]][m[:, 0]] == 1) & (degrees[names[1]][m[:, 1]] == 1)).sum())
    return dict(scope='selected_pairs_only_F_inliers', pairs=len(pairs), images=len(degrees),
                inlier_matches=edges, keypoint_degree_histogram={str(i): int(v) for i, v in enumerate(histogram)},
                keypoints_with_multiple_neighbors=int(histogram[2:].sum()), isolated_two_view_matches=isolated,
                isolated_two_view_fraction=isolated/edges if edges else None)


def write_index(root, index):
    index['pairs'] = dict(sorted(index['pairs'].items()))
    atomic_json(root/'index.json', index)
    # CSV is derived; NPZ/index remain authoritative and are resumable.
    import io
    buffer = io.StringIO(newline='')
    columns = ['image0', 'image1', 'keypoints0', 'keypoints1', 'forward_candidates', 'backward_candidates', 'matches', 'inliers', 'inlier_ratio', 'status', 'match_file']
    writer = csv.DictWriter(buffer, fieldnames=columns)
    writer.writeheader()
    writer.writerows({k: row[k] for k in columns} for row in index['pairs'].values())
    # Atomic JSON helper is for JSON only; use a separate atomic CSV publication.
    import os, tempfile
    fd, tmp = tempfile.mkstemp(prefix='.stats_', dir=root)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='') as f:
            f.write(buffer.getvalue())
        os.replace(tmp, root/'statistics.csv')
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def run(args, *, backend_factory=None, verifier=None):
    banks = FeatureBank(args.features_dir)
    if args.images_dir is None:
        args.images_dir = Path(banks.config['config']['images_dir']).resolve()
    if args.output_dir.is_relative_to(args.images_dir):
        raise ValueError('Match output must be outside the image directory')
    pairs = collect_pairs(banks.rows, args.pairs_file, args.overlap, args.max_pairs, args.pairing_mode)
    for name in sorted({n for pair in pairs for n in pair}):
        bank = banks.get(name)
        source_path(args.images_dir, name)
    selection = dict(mode='file' if args.pairs_file else args.pairing_mode, overlap=args.overlap,
                     offsets=sequential_offsets(len(banks.rows), args.overlap, args.pairing_mode) if not args.pairs_file else None,
                     pairs_file=str(args.pairs_file) if args.pairs_file else None,
                     max_pairs=args.max_pairs, selected_pairs=len(pairs))
    print(f'Pairs: {len(pairs)} | pairing: {selection["mode"]} | feature set: {banks.set_id} | output: {args.output_dir}', flush=True)
    if selection['offsets'] is not None:
        print(f'Index offsets (clipped to bank size): {selection["offsets"]}', flush=True)
    if args.dry_run:
        for names in pairs[:12]:
            print('  '.join(names))
        return 0
    if backend_factory is None:
        from .roma import RoMaMatcher
        backend_factory = RoMaMatcher
    if verifier is None:
        import cv2
        verifier, geometry_version = verify_fundamental, cv2.__version__
    else:
        geometry_version = 'injected_verifier'
    print('Preparing RoMa provenance (local checkpoints only)...', flush=True)
    backend = backend_factory(weights_dir=args.weights_dir, roma_repo_dir=args.roma_repo_dir,
        device=args.device, coarse_res=args.coarse_res, upsample_res=args.upsample_res, num_threads=args.num_threads)
    config = dict(schema_version=MATCH_SCHEMA, algorithm='fixed_ids_bidirectional_v1', feature_set_id=banks.set_id,
        images_dir=str(args.images_dir), backend=backend.signature,
        max_assignment_distance=args.max_assignment_distance, nn_ratio=args.nn_ratio,
        min_confidence=args.min_confidence, max_cycle_error=args.max_cycle_error,
        ransac_threshold=args.ransac_threshold, ransac_confidence=args.ransac_confidence,
        ransac_max_iters=args.ransac_max_iters, min_pair_inliers=args.min_pair_inliers,
        geometry='F_USAC_MAGSAC_on_fixed_coordinates', opencv_version=geometry_version,
        matching_code_sha256=sha256(Path(__file__).with_name('matching.py').read_bytes()).hexdigest())
    with output_lock(args.output_dir):
        run_id, index = prepare_output(args.output_dir, config)
        atomic_json(args.output_dir/'pairs_used.json', [list(p) for p in pairs])
        # Pair selection changes only which results are requested, not the
        # algorithm for a given pair. Keep it outside immutable run_config so
        # changing modes can reuse existing pair NPZs without recomputation.
        atomic_json(args.output_dir/'pairing_config.json', selection)
        LOG.setLevel(logging.INFO)
        LOG.propagate = False
        handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(args.output_dir/'matching.log', encoding='utf-8')]
        for h in handlers:
            h.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
            LOG.addHandler(h)
        status, started, counts = 'failed', perf_counter(), dict(matched=0, cached=0)
        try:
            for i, names in enumerate(pairs, 1):
                t = perf_counter()
                result, state = process_pair(args, backend, banks, names, run_id, verifier)
                counts[state] += 1
                m, nin = len(result.matches), int(result.inliers.sum())
                row = dict(image0=names[0], image1=names[1], keypoints0=result.metadata['keypoint_counts'][0],
                    keypoints1=result.metadata['keypoint_counts'][1], **result.metadata['stats'], matches=m,
                    inliers=nin, inlier_ratio=nin/m if m else 0.,
                    status='ok' if nin >= args.min_pair_inliers else 'insufficient_inliers',
                    content_id=result.content_id, match_file=match_path(args.output_dir, *names).relative_to(args.output_dir).as_posix())
                index['pairs'][pair_id(*names)] = row
                LOG.info('[%d/%d] %s <-> %s | %s | reciprocal=%d inliers=%d | %.2fs',
                         i, len(pairs), *names, state, m, nin, perf_counter()-t)
            atomic_json(args.output_dir/'association_stats.json', association_report(args.output_dir, pairs))
            status = 'ok'
            LOG.info('Finished: matched=%d cached=%d', counts['matched'], counts['cached'])
        except KeyboardInterrupt:
            status = 'interrupted'
            LOG.warning('Interrupted; completed pairs can be resumed unchanged')
            raise
        except Exception:
            LOG.exception('Matching stopped; existing feature banks and completed pair files are unchanged')
            raise
        finally:
            try:
                write_index(args.output_dir, index)
                atomic_json(args.output_dir/'last_run.json', dict(status=status, selected_pairs=len(pairs), **counts,
                            elapsed_seconds=perf_counter()-started))
            finally:
                for h in handlers:
                    LOG.removeHandler(h)
                    h.close()
    return 0


def main(argv=None):
    try:
        return run(parse_args(argv))
    except KeyboardInterrupt:
        print('Interrupted; completed match files can be resumed unchanged.', file=sys.stderr)
        return 130
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
