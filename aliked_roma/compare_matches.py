"""Compare whole-image matching information using immutable result banks.

The default compares whole images. Explicit user-provided evaluation masks can
compare different extraction masks on the same target. Cross-bank proximity
requires nearby coordinates in BOTH images; saved IDs are compared only within
their own bank. All geometry here is initial F screening, not 3D truth.
"""
from __future__ import annotations

import argparse
import csv
from hashlib import sha256
from io import BytesIO
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = Path(__file__).resolve().parent.name

from .extract import atomic_json, read_mask, resolve_masks
from .features import digest_json
from .match import FeatureBank, collect_pairs, verified_image_bytes
from .match_store import load_matches


def occupied_cells(points, image_size, cell_size):
    """Whole-image cell IDs, including partial right/bottom cells."""
    columns = math.ceil(image_size[0]/cell_size)
    xy = np.floor(np.asarray(points)/cell_size).astype(np.int64)
    return set((xy[:, 1]*columns+xy[:, 0]).tolist())


def point_membership(points, mask):
    """Test stored COLMAP coordinates against an explicit evaluation mask."""
    points = np.asarray(points)
    if mask is None:
        return np.ones(len(points), bool)
    if mask.ndim != 2 or mask.dtype != np.bool_:
        raise ValueError('Evaluation mask must be bool [H,W]')
    valid = (np.isfinite(points).all(1) & (points >= 0).all(1)
             & (points < [mask.shape[1], mask.shape[0]]).all(1))
    rows = np.flatnonzero(valid)
    pixels = np.floor(points[rows]).astype(np.int64)
    result = np.zeros(len(points), bool)
    result[rows] = mask[pixels[:, 1], pixels[:, 0]]
    return result


def match_membership(matches, points0, points1, mask0, mask1):
    """Both endpoints must belong to the same declared evaluation scope."""
    return (point_membership(points0[matches[:, 0]], mask0)
            & point_membership(points1[matches[:, 1]], mask1))


def active_mask_cells(mask, cell_size):
    """Cells containing any keep pixel; partial border cells are included."""
    h, w = mask.shape
    rows, columns = math.ceil(h/cell_size), math.ceil(w/cell_size)
    padded = np.pad(mask, ((0, rows*cell_size-h), (0, columns*cell_size-w)))
    occupied = padded.reshape(rows, cell_size, columns, cell_size).any(axis=(1, 3))
    return set(np.flatnonzero(occupied).tolist())


def covered_correspondences(query, reference, radius):
    """Directional proximity test; no one-to-one or physical-point claim.

    [N,4] rows are x0,y0,x1,y1 in ORIGINAL image pixels. A query is covered
    only if one reference row is within radius in BOTH endpoint images.
    """
    from scipy.spatial import cKDTree
    query, reference = np.asarray(query), np.asarray(reference)
    if query.ndim != 2 or query.shape[1] != 4 or reference.ndim != 2 or reference.shape[1] != 4:
        raise ValueError('Correspondences must be [N,4] original-image coordinates')
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError('Proximity radius must be finite and positive')
    covered = np.zeros(len(query), bool)
    if not len(query) or not len(reference):
        return covered
    neighbors = cKDTree(reference[:, :2]).query_ball_point(query[:, :2], radius)
    for i, rows in enumerate(neighbors):
        if rows:
            delta = reference[rows, 2:].astype(np.float64)-query[i, 2:]
            covered[i] = bool((np.sum(delta*delta, axis=1) <= radius*radius).any())
    return covered


def closed_triplets(pair_matches):
    """Enumerate all image triangles and close their own bank's row IDs."""
    adjacency = {}
    for a, b in pair_matches:
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
    result = []
    for a, b in sorted(pair_matches):
        for c in sorted(adjacency[a] & adjacency[b]):
            if c <= b:
                continue
            ab, ac, bc = (dict(map(tuple, pair_matches[key])) for key in ((a, b), (a, c), (b, c)))
            rows = [[i, ab[i], ac[i]] for i in sorted(ab.keys() & ac.keys()) if bc.get(ab[i]) == ac[i]]
            result.append(((a, b, c), np.asarray(rows, np.uint32).reshape(-1, 3)))
    return result


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def plot_summary(output, methods, pair_rows, totals, novelty, masked=False):
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/roma_comparison_mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for label in methods:
        rows = [r for r in pair_rows if r['method'] == label]
        axes[0, 0].plot(np.arange(1, len(rows)+1), [r['f_inliers'] for r in rows], 'o-', label=label)
    axes[0, 0].set(title='F inliers inside supplied masks' if masked else 'Initial F inliers for the same image pairs',
                   xlabel='Pair index', ylabel='Correspondences')
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].bar(list(methods), [totals[m]['closed_three_view_correspondences'] for m in methods])
    axes[0, 1].set(title='Closed three-view correspondences', ylabel='Count across image triangles')
    x = np.arange(3)
    bar_width = .8/len(methods)
    for i, label in enumerate(methods):
        offset = (i-(len(methods)-1)/2)*bar_width
        axes[1, 0].bar(x+offset, [totals[label]['coverage'][str(g)]['fraction']*100 for g in (32, 64, 128)],
                       width=bar_width, label=label)
    axes[1, 0].set(xticks=x, xticklabels=['32 px', '64 px', '128 px'],
                   title='Mask-cell coverage by matched observations' if masked else 'Whole-image coverage by matched observations',
                   ylabel='Occupied mask cells (%)' if masked else 'Occupied image cells (%)')
    axes[1, 0].legend(fontsize=8)
    candidates = list(methods)[1:]
    rows = [next(r for r in novelty if r['method'] == label and r['radius_px'] == 3) for label in candidates]
    x = np.arange(len(rows))
    axes[1, 1].bar(x-.18, [r['candidate_without_baseline_neighbor'] for r in rows], width=.36, label='Candidate without baseline neighbor')
    axes[1, 1].bar(x+.18, [r['baseline_without_candidate_neighbor'] for r in rows], width=.36, label='Baseline without candidate neighbor')
    axes[1, 1].set(xticks=x, xticklabels=candidates, title='Both endpoints within 3 px: directional gaps', ylabel='Correspondence edges')
    axes[1, 1].legend(fontsize=8)
    for ax in axes.ravel():
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    fig.savefig(output/'comparison_charts.png', dpi=160)
    plt.close(fig)


def plot_matched_points(output, methods, names, banks, degrees, cells, images, evaluation_masks=None):
    """Show every matched observation; cyan identifies new 64-px cells."""
    width, header = 640, 48
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14)
    baseline = next(iter(methods))
    pages = []
    for start in range(0, len(names), 3):
        selected = names[start:start+3]
        heights = [round(images[n].height*width/images[n].width) for n in selected]
        canvas = Image.new('RGB', (width*len(methods), sum(h+header for h in heights)), '#171b20')
        top = 0
        for name, height in zip(selected, heights):
            image = images[name]
            for column, label in enumerate(methods):
                frame = Image.new('RGB', (width, height+header), '#171b20')
                preview = image.resize((width, height), Image.Resampling.LANCZOS)
                if evaluation_masks is not None:
                    mask_image = Image.fromarray(evaluation_masks[name].astype(np.uint8)*255).resize(
                        (width, height), Image.Resampling.NEAREST)
                    dimmed = Image.blend(preview, Image.new('RGB', preview.size), .6)
                    preview = Image.composite(preview, dimmed, mask_image)
                    preview.paste('#39e58c', mask=mask_image.filter(ImageFilter.FIND_EDGES))
                frame.paste(preview, (0, header))
                draw = ImageDraw.Draw(frame)
                points = banks[label].get(name).keypoints[degrees[label][name] > 0]
                new_cells = cells[label][name][64]-cells[baseline][name][64]
                draw.text((8, 3), f'{label}: {len(points)} matched observations; +{len(new_cells)} cells', font=font, fill='white')
                draw.text((8, 25), name, font=font, fill='#c1cad3')
                columns = math.ceil(image.width/64)
                for px, py in points:
                    cid = int(py//64)*columns+int(px//64)
                    color = '#18e5ee' if cid in new_cells else '#ffc541'
                    x, y = float(px)*width/image.width, float(py)*height/image.height+header
                    draw.ellipse((x-.9, y-.9, x+.9, y+.9), fill=color)
                canvas.paste(frame, (column*width, top))
            top += height+header
        filename = f'matched_observations_{start//3+1:02d}.jpg'
        canvas.save(output/filename, quality=95)
        pages.append(filename)
    return pages


def run(args):
    methods = {args.baseline_label: (args.baseline_features, args.baseline_matches)}
    for label, features, matches in args.candidate:
        if label in methods:
            raise ValueError(f'Duplicate method label: {label}')
        methods[label] = (Path(features).resolve(), Path(matches).resolve())
    if len(methods) < 2:
        raise ValueError('Supply at least one candidate')
    banks = {label: FeatureBank(paths[0]) for label, paths in methods.items()}
    baseline = next(iter(methods))
    pairs = collect_pairs(banks[baseline].rows, args.pairs_file)
    names = sorted({name for pair in pairs for name in pair})
    configs, references = {}, {}
    for label, (features, matches) in methods.items():
        run_config = json.loads((matches/'run_config.json').read_text())
        if digest_json(run_config['config']) != run_config['run_id']:
            raise ValueError(f'Invalid matching configuration: {label}')
        if run_config['config']['feature_set_id'] != banks[label].set_id:
            raise ValueError(f'Matching run refers to another feature bank: {label}')
        configs[label] = run_config
        comparable = {k: v for k, v in run_config['config'].items() if k != 'feature_set_id'}
        if label == baseline:
            settings = comparable
        elif comparable != settings:
            raise ValueError(f'Matching settings/backend differ from baseline: {label}')
        references[label] = dict(features_dir=str(features), matches_dir=str(matches),
                                 feature_set_id=banks[label].set_id, matching_run_id=run_config['run_id'])
    # Source identities always agree. Feature-mask changes require an explicit
    # common evaluation mask; the default detector comparison stays strict.
    images, evaluation_masks, evaluation_sources = {}, None, {}
    evaluation_dir = getattr(args, 'evaluation_masks_dir', None)
    mask_paths = resolve_masks(evaluation_dir, names) if evaluation_dir is not None else {}
    if evaluation_dir is not None:
        evaluation_masks = {}
    image_root = Path(configs[baseline]['config']['images_dir'])
    for name in names:
        ref = banks[baseline].get(name)
        raw = verified_image_bytes(image_root, ref)
        with Image.open(BytesIO(raw)) as image:
            images[name] = image.convert('RGB')
        if evaluation_masks is not None:
            mask_raw = mask_paths[name].read_bytes()
            evaluation_masks[name] = read_mask(mask_raw, images[name].size, args.evaluation_mask_threshold)
            evaluation_sources[name] = dict(path=str(mask_paths[name]), sha256=sha256(mask_raw).hexdigest(),
                                           keep_pixels=int(evaluation_masks[name].sum()))
        for label in methods:
            other = banks[label].get(name)
            if other.source_sha256 != ref.source_sha256 or not np.array_equal(other.image_size, ref.image_size):
                raise ValueError(f'Source images differ: {label}, {name}')
            if evaluation_masks is None and other.mask_sha256 != ref.mask_sha256:
                raise ValueError(f'Feature masks differ: {label}, {name}; declare --evaluation-masks-dir to compare mask changes')
    pair_rows, image_rows, triple_rows, coordinates, degrees, cells, totals = [], [], [], {}, {}, {}, {}
    for label, (_, match_root) in methods.items():
        bank = banks[label]
        degrees[label] = {name: np.zeros(len(bank.get(name).keypoints), np.uint32) for name in names}
        coordinates[label], selected_pairs = {}, {}
        for a, b in pairs:
            result = load_matches(match_root, a, b)
            f0, f1 = bank.get(a), bank.get(b)
            result.bind(f0, f1, configs[label]['run_id'])
            eligible = (match_membership(result.matches, f0.keypoints, f1.keypoints,
                        evaluation_masks[a], evaluation_masks[b]) if evaluation_masks is not None
                        else np.ones(len(result.matches), bool))
            selected = result.matches[result.inliers & eligible]
            selected_pairs[a, b] = selected
            coordinates[label][a, b] = np.column_stack((f0.keypoints[selected[:, 0]], f1.keypoints[selected[:, 1]]))
            for side, name in enumerate((a, b)):
                degrees[label][name][selected[:, side]] += 1
            pair_rows.append(dict(method=label, image0=a, image1=b, reciprocal=int(eligible.sum()),
                                  f_inliers=len(selected), all_reciprocal=len(result.matches),
                                  all_f_inliers=int(result.inliers.sum()), content_id=result.content_id))
        triplets = closed_triplets(selected_pairs)
        for (a, b, c), tracks in triplets:
            triple_rows.append(dict(method=label, image0=a, image1=b, image2=c, closed_correspondences=len(tracks)))
        cells[label] = {}
        for name in names:
            feature = bank.get(name)
            d = degrees[label][name]
            points = feature.keypoints[d > 0]
            cells[label][name] = {g: occupied_cells(points, feature.image_size, g) for g in (32, 64, 128)}
            image_rows.append(dict(method=label, image=name, detections=len(d), matched_observations=int((d > 0).sum()),
                                   detections_in_scope=int(point_membership(feature.keypoints,
                                       evaluation_masks[name] if evaluation_masks is not None else None).sum()),
                                   observations_with_two_or_more_neighbors=int((d >= 2).sum()),
                                   **{f'occupied_{g}px_cells': len(cells[label][name][g]) for g in (32, 64, 128)}))
        totals[label] = dict(pairs=len(pairs),
            reciprocal=sum(r['reciprocal'] for r in pair_rows if r['method'] == label),
            f_inliers=sum(r['f_inliers'] for r in pair_rows if r['method'] == label),
            all_reciprocal=sum(r['all_reciprocal'] for r in pair_rows if r['method'] == label),
            all_f_inliers=sum(r['all_f_inliers'] for r in pair_rows if r['method'] == label),
            detections_all=sum(r['detections'] for r in image_rows if r['method'] == label),
            detections_in_scope=sum(r['detections_in_scope'] for r in image_rows if r['method'] == label),
            matched_observations=sum(int((d > 0).sum()) for d in degrees[label].values()),
            observations_with_two_or_more_neighbors=sum(int((d >= 2).sum()) for d in degrees[label].values()),
            closed_three_view_correspondences=sum(len(t) for _, t in triplets), coverage={})
        baseline_counts = {(r['image0'], r['image1']): r['f_inliers'] for r in pair_rows if r['method'] == baseline}
        totals[label]['pairs_with_more_f_inliers_than_baseline'] = sum(
            r['f_inliers'] > baseline_counts[r['image0'], r['image1']] for r in pair_rows if r['method'] == label)
        for g in (32, 64, 128):
            occupied = sum(len(cells[label][n][g]) for n in names)
            possible = (sum(len(active_mask_cells(evaluation_masks[n], g)) for n in names)
                        if evaluation_masks is not None else
                        sum(math.ceil(images[n].width/g)*math.ceil(images[n].height/g) for n in names))
            totals[label]['coverage'][str(g)] = dict(occupied_cells=occupied, total_image_cells=possible,
                                                    fraction=occupied/possible if possible else 0.,
                gained_vs_baseline=sum(len(cells[label][n][g]-cells[baseline][n][g]) for n in names),
                lost_vs_baseline=sum(len(cells[baseline][n][g]-cells[label][n][g]) for n in names))
    novelty_pairs, novelty_totals = [], []
    for label in list(methods)[1:]:
        for radius in (1, 3, 5):
            rows = []
            for a, b in pairs:
                query, reference = coordinates[label][a, b], coordinates[baseline][a, b]
                row = dict(method=label, image0=a, image1=b, radius_px=radius,
                           candidate_without_baseline_neighbor=int((~covered_correspondences(query, reference, radius)).sum()),
                           baseline_without_candidate_neighbor=int((~covered_correspondences(reference, query, radius)).sum()))
                rows.append(row)
            novelty_pairs.extend(rows)
            novelty_totals.append(dict(method=label, radius_px=radius,
                candidate_without_baseline_neighbor=sum(r['candidate_without_baseline_neighbor'] for r in rows),
                baseline_without_candidate_neighbor=sum(r['baseline_without_candidate_neighbor'] for r in rows)))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/'image_list.txt').write_text('\n'.join(names)+'\n', encoding='utf-8')
    (args.output_dir/'pairs.txt').write_text(''.join(f'{a} {b}\n' for a, b in pairs), encoding='utf-8')
    for filename, rows in [('pair_statistics.csv', pair_rows), ('image_statistics.csv', image_rows),
                           ('triplet_statistics.csv', triple_rows), ('proximity_by_pair.csv', novelty_pairs),
                           ('proximity_summary.csv', novelty_totals)]:
        write_csv(args.output_dir/filename, rows)
    summary = dict(sources=references, matching_config_identical_except_feature_bank=True,
        matching_settings=settings, pairs=[list(p) for p in pairs], images=names, totals=totals,
        evaluation_scope='provided_masks_both_endpoints' if evaluation_masks is not None else 'whole_image',
        evaluation_masks=evaluation_sources,
        proximity_summary=novelty_totals,
        definitions=dict(matched_observations='Unique (image_name, saved_row_id) with at least one F-inlier edge',
                         coverage=('Uniform cells containing keep pixels; matched observations restricted to provided masks'
                                   if evaluation_masks is not None else
                                   'Whole-image uniform cells containing any F-matched observation; no region annotations'),
                         proximity='Directional test: the same reference edge must be within radius at BOTH image endpoints',
                         closed='All three pair edges are F inliers with closing IDs; summed across image triangles',
                         limitations='Proximity and F inliers do not prove independent physical points or 3D accuracy'))
    atomic_json(args.output_dir/'summary.json', summary)
    plot_summary(args.output_dir, methods, pair_rows, totals, novelty_totals, masked=evaluation_masks is not None)
    pages = plot_matched_points(args.output_dir, methods, names, banks, degrees, cells, images, evaluation_masks)
    lines = ['# 固定点库+RoMa：'+('指定掩膜内' if evaluation_masks is not None else '整图')+'匹配信息对照', '',
             f'同一组 {len(names)} 张图、{len(pairs)} 对配对；RoMa 模型、权重、源码、分辨率和全部匹配参数完全相同。',
             ('提取掩膜可以不同；所有方案在同一份评估掩膜内比较。点预算与选点策略沿用各方案配置。'
              if evaluation_masks is not None else
              '使用相同源图与既有 keep-mask；各检测器沿用其点预算和选点策略，实际检测点数见逐图表。'),
             ('使用明确指定的同一份评估掩膜，匹配两端均需在白色区域内。旧方案的背景对应不计入目标对照。'
              if evaluation_masks is not None else '本报告不读取、不绘制、不使用任何人工区域标注。'), '',
             '| 方法 | 双向对应 | 初始 F 内点 | 唯一匹配观测 | 复用于至少两个邻图的观测 | 三图闭环对应 |',
             '|---|---:|---:|---:|---:|---:|']
    if evaluation_masks is not None:
        lines[6:6] = [f'评估掩膜：`{evaluation_dir}`；白色阈值 {args.evaluation_mask_threshold}。文件哈希与保留面积见 summary.json。', '']
    for label, t in totals.items():
        lines.append(f"| {label} | {t['reciprocal']} | {t['f_inliers']} | {t['matched_observations']} | "
                     f"{t['observations_with_two_or_more_neighbors']} | {t['closed_three_view_correspondences']} |")
    lines += ['', '当前样本的替换结果：']
    for label in list(methods)[1:]:
        t, ref = totals[label], totals[baseline]
        count_change = t['f_inliers']/ref['f_inliers']-1 if ref['f_inliers'] else None
        closed_change = (t['closed_three_view_correspondences']/ref['closed_three_view_correspondences']-1
                         if ref['closed_three_view_correspondences'] else None)
        if count_change is not None and closed_change is not None:
            lines.append(f"- {label}：F 内点变化 {count_change:+.2%}，三图闭环变化 {closed_change:+.2%}；"
                         f"有 {t['pairs_with_more_f_inliers_than_baseline']}/{len(pairs)} 对图像的 F 内点数超过基线。")
    coverage_label = '掩膜格' if evaluation_masks is not None else '整图格'
    lines += ['', '匹配观测按同一图内保存的 ID 去重；三图闭环按图像三元组累计，不等同于已重建三维点。', '',
              f'| 方法 | 32px {coverage_label}覆盖率 | 64px {coverage_label}覆盖率 | 128px {coverage_label}覆盖率 |', '|---|---:|---:|---:|']
    for label, t in totals.items():
        rates = [f"{t['coverage'][str(g)]['fraction']:.2%}" for g in (32, 64, 128)]
        lines.append('| '+label+' | '+' | '.join(rates)+' |')
    lines += ['', '| 方法 | 两端均以 3px 判近邻：候选未被基线覆盖的对应 | 基线未被候选覆盖的对应 | 新增 64px 图像格 | 丢失 64px 图像格 |',
              '|---|---:|---:|---:|---:|']
    for label in list(methods)[1:]:
        r = next(r for r in novelty_totals if r['method'] == label and r['radius_px'] == 3)
        c = totals[label]['coverage']['64']
        lines.append(f"| {label} | {r['candidate_without_baseline_neighbor']} | {r['baseline_without_candidate_neighbor']} | "
                     f"{c['gained_vs_baseline']} | {c['lost_vs_baseline']} |")
    lines += ['', '近邻差异只表示两端像素位置的互补性，不证明这些对应是新的物理点；不同检测器的行 ID 没有直接比较。',
              '因此应同时看新增和丢失：候选可以补到基线未覆盖的位置，但这与替换后总体匹配量、复用和闭环是否增加是不同问题。',
              ('1px、3px、5px 的全部近邻统计保存在 proximity_summary.csv。覆盖率分母为包含白色保留像素的格子。'
               if evaluation_masks is not None else
               '1px、3px、5px 的全部近邻统计保存在 proximity_summary.csv。整图格覆盖率的分母包含全部图像格，边缘格按一格计。'), '',
              '[数量、闭环、覆盖与近邻差异图](comparison_charts.png)', '',
              '整图叠加显示全部参与 F 内点匹配的观测，没有抽样。黄色为匹配观测，青色为候选在基线未占用的 64px 图像格中的观测：', '']
    lines.extend(f'- [整图匹配观测对照 {i+1}]({p})' for i, p in enumerate(pages))
    lines += ['', '[逐对统计](pair_statistics.csv) · [逐图统计](image_statistics.csv) · [三图统计](triplet_statistics.csv) · [近邻统计](proximity_summary.csv) · [配置与定义](summary.json)', '',
              '范围仅限当前样本与筛选设置；初始 F 内点、闭环和空间覆盖不能代替标定几何、三角化或有真值的精度评估。', '']
    (args.output_dir/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-features', type=Path, required=True)
    p.add_argument('--baseline-matches', type=Path, required=True)
    p.add_argument('--baseline-label', default='ALIKED+RoMa', help='Label for the reference bank')
    p.add_argument('--candidate', nargs=3, action='append', required=True, metavar=('LABEL', 'FEATURES', 'MATCHES'))
    p.add_argument('--pairs-file', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--evaluation-masks-dir', type=Path,
                   help='Explicit common evaluation masks; both endpoints must be inside. Allows feature-mask changes.')
    p.add_argument('--evaluation-mask-threshold', type=int, default=128)
    args = p.parse_args(argv)
    for name in ('baseline_features', 'baseline_matches', 'pairs_file', 'output_dir'):
        setattr(args, name, getattr(args, name).resolve())
    if not 1 <= args.evaluation_mask_threshold <= 255:
        p.error('--evaluation-mask-threshold must be in [1,255]')
    if args.evaluation_masks_dir is not None:
        args.evaluation_masks_dir = args.evaluation_masks_dir.resolve()
    inputs = [args.baseline_features, args.baseline_matches]+[Path(path).resolve() for row in args.candidate for path in row[1:]]
    if args.evaluation_masks_dir is not None:
        inputs.append(args.evaluation_masks_dir)
    if any(args.output_dir.is_relative_to(root) or root.is_relative_to(args.output_dir) for root in inputs):
        p.error('Output must be separate from immutable feature and match banks')
    run(args)


if __name__ == '__main__':
    main()
