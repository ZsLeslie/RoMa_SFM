"""Plot fixed-ID pair matches and one available three-view closed association.

python -m aliked_roma.visualize --features-dir BANK --matches-dir MATCHES
Or run python visualize.py from the aliked_roma directory. No torch/GPU needed.
"""
from __future__ import annotations

import argparse
import csv
from functools import lru_cache
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = Path(__file__).resolve().parent.name

from .extract import source_path
from .match import FeatureBank
from .match_store import canonical_pair, load_matches, pair_id


def spatial_sample(points, rows, width, height, limit, seed=42):
    """Deterministic grid-balanced sampling; does not favor high confidence."""
    rows = np.asarray(rows, dtype=np.int64)
    if len(rows) <= limit:
        return rows
    cells = np.floor(points[rows] / [width/12, height/8]).astype(int)
    groups = {}
    for index in np.random.default_rng(seed).permutation(len(rows)):
        groups.setdefault(tuple(cells[index]), []).append(int(rows[index]))
    selected = []
    while len(selected) < limit:
        for key in sorted(groups):
            if groups[key]:
                selected.append(groups[key].pop())
                if len(selected) == limit:
                    break
    return np.asarray(selected, np.int64)


class Visualizer:
    def __init__(self, args):
        # Matplotlib is used as a scientific plotter; the original photos and
        # saved data remain untouched. Keep its cache in a writable temp path.
        os.environ.setdefault('MPLCONFIGDIR', '/tmp/roma_visualization_mpl')
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
        self.plt = plt
        self.norm = Normalize(vmin=0., vmax=1.)
        self.cmap = plt.get_cmap('viridis')
        self.args = args
        self.banks = FeatureBank(args.features_dir)
        config = json.loads((args.matches_dir/'run_config.json').read_text())
        self.images_dir = args.images_dir or Path(config['config']['images_dir'])

    @lru_cache(maxsize=6)
    def image(self, name):
        bank = self.banks.get(name)
        data = source_path(self.images_dir, name).read_bytes()
        if sha256(data).hexdigest() != bank.source_sha256:
            raise ValueError(f'Source image differs from fixed keypoint bank: {name}')
        with Image.open(BytesIO(data)) as f:
            rgb = np.asarray(f.convert('RGB')).copy()
        if rgb.shape[:2] != (bank.image_size[1], bank.image_size[0]):
            raise ValueError(f'Image size mismatch: {name}')
        return rgb

    def pair(self, names):
        a, b = canonical_pair(*names)
        result = load_matches(self.args.matches_dir, a, b)
        f0, f1 = self.banks.get(a), self.banks.get(b)
        result.bind(f0, f1)
        return result, f0, f1

    def background(self, ax, names):
        offsets, x, max_height = [], 0, 0
        for name in names:
            rgb = self.image(name)
            h, w = rgb.shape[:2]
            offsets.append(x)
            # Extent uses corner-origin coordinates, so image pixel centers
            # are 0.5 and agree directly with saved COLMAP/RoMa coordinates.
            ax.imshow(rgb, extent=(x, x+w, h, 0), interpolation='bilinear')
            x += w+35
            max_height = max(max_height, h)
        ax.set(xlim=(0, x-35), ylim=(max_height, 0))
        ax.set_aspect('equal')
        ax.axis('off')
        return offsets

    def draw_pair(self, ax, names, number):
        from matplotlib.collections import LineCollection
        result, f0, f1 = self.pair(names)
        offsets = self.background(ax, names)
        p0, p1 = f0.keypoints[result.matches[:, 0]], f1.keypoints[result.matches[:, 1]]
        selected = spatial_sample(p0, np.flatnonzero(result.inliers), *f0.image_size, self.args.max_lines)
        for rows, colors, linewidth in [(selected, self.cmap(self.norm(result.scores[selected])), .65),
                                        (np.flatnonzero(~result.inliers), 'red', 1.2)]:
            q0, q1 = p0[rows], p1[rows]+[offsets[1], 0]
            if not len(rows):
                continue
            ax.add_collection(LineCollection(np.stack((q0, q1), axis=1), colors=colors,
                                              linewidths=linewidth, alpha=.8))
            ax.scatter(q0[:, 0], q0[:, 1], s=9, c=colors, linewidths=.35, edgecolors='white', zorder=3)
            ax.scatter(q1[:, 0], q1[:, 1], s=9, c=colors, linewidths=.35, edgecolors='white', zorder=3)
        ni, no = int(result.inliers.sum()), int((~result.inliers).sum())
        ax.set_title(f'{number}. {names[0]}  <->  {names[1]}\n'
                     f'F inliers: {ni}/{len(result.matches)} | '
                     f'displayed: {len(selected)} spatial samples + {no} F outliers (red)', fontsize=11, pad=8)
        return dict(images=list(names), matches=len(result.matches), inliers=ni,
                    displayed_inliers=len(selected), displayed_outliers=no)

    def colorbar(self, fig):
        from matplotlib.cm import ScalarMappable
        axis = fig.add_axes([.30, .028, .40, .012])
        bar = fig.colorbar(ScalarMappable(norm=self.norm, cmap=self.cmap), cax=axis, orientation='horizontal')
        bar.set_label('RoMa confidence (minimum along forward / backward paths)', fontsize=9)
        bar.ax.tick_params(labelsize=8)

    def find_closed_tracks(self, pairs):
        adjacency = {}
        for a, b in pairs:
            adjacency.setdefault(a, set()).add(b)
            adjacency.setdefault(b, set()).add(a)
        checked = 0
        for a, b in pairs:
            for c in sorted(adjacency[a] & adjacency[b]):
                if c <= b:
                    continue
                checked += 1
                maps = []
                for names in [(a, b), (a, c), (b, c)]:
                    r, _, _ = self.pair(names)
                    maps.append({int(i): (int(j), float(score)) for (i, j), score in zip(r.matches[r.inliers], r.scores[r.inliers])})
                ab, ac, bc = maps
                rows = []
                for i in sorted(ab.keys() & ac.keys()):
                    j, k = ab[i][0], ac[i][0]
                    if j in bc and bc[j][0] == k:
                        rows.append([i, j, k, min(ab[i][1], ac[i][1], bc[j][1])])
                if rows:
                    return (a, b, c), np.asarray(rows, dtype=np.float64)
                if checked >= 100:
                    return None, None
        return None, None

    def draw_tracks(self, names, tracks):
        from matplotlib.collections import LineCollection
        f = [self.banks.get(name) for name in names]
        points = [bank.keypoints[tracks[:, side].astype(int)] for side, bank in enumerate(f)]
        candidates = spatial_sample(points[0], np.arange(len(tracks)), *f[0].image_size,
                                    min(len(tracks), self.args.max_tracks*8))
        selected = []
        for row in candidates:
            # Keep labels readable in ALL views, including near grid borders.
            if selected and any((np.linalg.norm(points[side][selected]-points[side][row], axis=1)
                                 < .05*min(f[side].image_size)).any() for side in range(3)):
                continue
            selected.append(int(row))
            if len(selected) == self.args.max_tracks:
                break
        selected = np.asarray(selected, dtype=np.int64)
        fig, ax = self.plt.subplots(figsize=(21, 6.5))
        offsets = self.background(ax, names)
        colors = self.plt.get_cmap('turbo')(np.linspace(.03, .97, len(selected)))
        for color, row in zip(colors, selected):
            xyz = np.array([points[side][row]+[offsets[side], 0] for side in range(3)])
            ax.add_collection(LineCollection(np.stack((xyz[:-1], xyz[1:]), axis=1), colors=[color], linewidths=1.1, alpha=.8))
            ax.scatter(xyz[:, 0], xyz[:, 1], c=[color], s=22, edgecolors='white', linewidths=.7, zorder=4)
            for point in xyz:
                ax.annotate(str(row+1), point, xytext=(4, -4), textcoords='offset points', color='white', fontsize=7,
                            bbox=dict(boxstyle='round,pad=.12', fc='black', alpha=.65, ec='none'), zorder=5)
        ax.set_title('Three-view closed correspondences (F inliers in all three pairs)\n'
                     f'{names[0]}  |  {names[1]}  |  {names[2]}\n'
                     f'{len(tracks)} closed correspondences; {len(selected)} spatial samples shown. Same number/color = same correspondence.',
                     fontsize=12, pad=10)
        fig.subplots_adjust(left=.005, right=.995, top=.86, bottom=.02)
        fig.savefig(self.args.output_dir/'three_view_correspondences.png', dpi=160, bbox_inches='tight')
        self.plt.close(fig)
        with (self.args.output_dir/'three_view_correspondences.csv').open('w', newline='') as out:
            writer = csv.writer(out)
            writer.writerow(['display_number', 'image0', 'keypoint_id0', 'image1', 'keypoint_id1', 'image2', 'keypoint_id2', 'min_confidence', 'shown'])
            for i, row in enumerate(tracks):
                writer.writerow([i+1, names[0], int(row[0]), names[1], int(row[1]), names[2], int(row[2]), float(row[3]), i in selected])
        return dict(images=list(names), total_closed_correspondences=len(tracks), displayed=len(selected),
                    image='three_view_correspondences.png', table='three_view_correspondences.csv')

    def run(self):
        args = self.args
        if args.pair:
            pairs = [canonical_pair(*args.pair)]
        else:
            pairs = sorted({canonical_pair(*p) for p in json.loads((args.matches_dir/'pairs_used.json').read_text())})
        if args.max_pairs:
            pairs = pairs[:args.max_pairs]
        if not pairs:
            raise ValueError('No match pairs to visualize')
        args.output_dir.mkdir(parents=True, exist_ok=True)
        manifest = dict(features_dir=str(args.features_dir), matches_dir=str(args.matches_dir),
                        images_dir=str(self.images_dir), pairs=[], three_view=None)
        for i, names in enumerate(pairs, 1):
            fig, ax = self.plt.subplots(figsize=(16, 6.8))
            summary = self.draw_pair(ax, names, i)
            self.colorbar(fig)
            fig.subplots_adjust(left=.01, right=.99, top=.91, bottom=.10)
            filename = f'pair_{i:02d}_{pair_id(*names)[:12]}.png'
            fig.savefig(args.output_dir/filename, dpi=150, bbox_inches='tight')
            self.plt.close(fig)
            summary['image'] = filename
            manifest['pairs'].append(summary)
            print(f'{filename}: {summary["inliers"]}/{summary["matches"]} F inliers', flush=True)
        n = min(len(pairs), 6)
        fig, axes = self.plt.subplots(n, 1, figsize=(16, 6.5*n), squeeze=False)
        for i, names in enumerate(pairs[:n], 1):
            self.draw_pair(axes[i-1, 0], names, i)
        self.colorbar(fig)
        fig.subplots_adjust(left=.01, right=.99, top=.98, bottom=.06, hspace=.18)
        fig.savefig(args.output_dir/'matches_overview.jpg', dpi=120, bbox_inches='tight', pil_kwargs={'quality': 94})
        self.plt.close(fig)
        names, tracks = self.find_closed_tracks(pairs)
        if names is not None:
            manifest['three_view'] = self.draw_tracks(names, tracks)
            print(f'three_view_correspondences.png: {len(tracks)} closed correspondences', flush=True)
        (args.output_dir/'visualization_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        return manifest


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--features-dir', type=Path, required=True)
    p.add_argument('--matches-dir', type=Path, required=True)
    p.add_argument('--images-dir', type=Path)
    p.add_argument('--output-dir', type=Path, help='Default: MATCHES/visualizations')
    p.add_argument('--pair', nargs=2, metavar=('IMAGE0', 'IMAGE1'))
    p.add_argument('--max-pairs', type=int, default=3, help='0 = all selected pairs; overview shows at most six')
    p.add_argument('--max-lines', type=int, default=100)
    p.add_argument('--max-tracks', type=int, default=24)
    args = p.parse_args(argv)
    if args.max_pairs < 0 or args.max_lines < 1 or args.max_tracks < 1:
        p.error('max-pairs must be >=0; max-lines and max-tracks must be positive')
    for name in ['features_dir', 'matches_dir', 'images_dir', 'output_dir']:
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).expanduser().resolve())
    args.output_dir = args.output_dir or args.matches_dir/'visualizations'
    if args.output_dir.is_relative_to(args.features_dir):
        p.error('Visualization output must be outside the immutable feature bank')
    try:
        Visualizer(args).run()
        return 0
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
