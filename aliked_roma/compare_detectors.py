"""Compare fixed detector banks on the previously annotated blade interiors.

Run with --prepare-only to write the exact four anchors and adjacent images,
then extract new banks using image_list.txt and rerun with --bank LABEL=DIR.
The original ALIKED bank, masks and annotations are read-only.
"""
from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = Path(__file__).resolve().parent.name

from .extract import atomic_json, read_mask, resolve_masks, source_path
from .match import FeatureBank


def roi_mask(polygons, image_size):
    """Exactly reproduce the diagnosis's PIL rasterization, including edges."""
    region = Image.new('L', image_size)
    draw = ImageDraw.Draw(region)
    width, height = image_size
    for polygon in polygons:
        draw.polygon([(x*width/960, y*height/540) for x, y in polygon], fill=255)
    return np.asarray(region) > 0


def roi_membership(points, polygons, image_size):
    pixels = np.floor(points).astype(np.int64)
    return roi_mask(polygons, image_size)[pixels[:, 1], pixels[:, 0]]


def panel(image, features, label, polygons=None, width=720):
    scale = width/image.width
    height = round(image.height*scale)
    canvas = Image.new('RGB', (width, height+48), '#171b20')
    canvas.paste(image.resize((width, height), Image.Resampling.LANCZOS), (0, 48))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 15)
    points = features.keypoints
    inside = (roi_membership(points, polygons, image.size)
              if polygons is not None else np.zeros(len(points), bool))
    text = f'{label} | total {len(points)}'
    if polygons is not None:
        text += f' | blade ROI {int(inside.sum())}'
    draw.text((9, 4), text, fill='white', font=font)
    draw.text((9, 26), features.image_name, fill='#bec8d0', font=font)
    for point, on_blade in zip(points, inside):
        x, y = point*scale
        y += 48
        radius = 1.8 if on_blade else .8
        draw.ellipse((x-radius, y-radius, x+radius, y+radius),
                     fill='#ff3864' if on_blade else '#ffc541')
    if polygons is not None:
        for polygon in polygons:
            xy = [(x*width/960, y*height/540+48) for x, y in polygon]
            draw.line(xy+[xy[0]], fill='#18d4ee', width=2)
    return canvas


def run(args):
    baseline = FeatureBank(args.baseline_dir)
    config = baseline.config['config']
    image_root = Path(config['images_dir'])
    mask_root = Path(config['masks_dir']) if config.get('masks_dir') else None
    diagnosis = json.loads((args.diagnosis_dir/'statistics.json').read_text())
    names = sorted(baseline.rows)
    anchors = sorted(diagnosis['manual_blade_roi_samples'], key=lambda r: r['bank_keypoints'])
    selected = set()
    groups = []
    for anchor in anchors:
        name = anchor['name']
        index = names.index(name)
        neighbors = names[max(0, index-args.neighbors):index+args.neighbors+1]
        selected.update(neighbors)
        groups.append(dict(image_id=anchor['image_id'], anchor=name,
                           previous_roi_count=anchor['bank_keypoints'], views=neighbors))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selection = dict(baseline_dir=str(args.baseline_dir), diagnosis_dir=str(args.diagnosis_dir),
                     neighbors_each_side=args.neighbors, images=sorted(selected), groups=groups,
                     roi_scope='manual blade-interior polygons on the four anchor images only')
    atomic_json(args.output_dir/'selection.json', selection)
    (args.output_dir/'image_list.txt').write_text('\n'.join(sorted(selected))+'\n')
    if args.prepare_only:
        print(json.dumps(selection, ensure_ascii=False, indent=2))
        return
    banks = {'ALIKED grid256': baseline}
    for spec in args.bank:
        label, path = spec.split('=', 1)
        if label in banks or not label:
            raise ValueError(f'Duplicate or empty bank label: {label!r}')
        banks[label] = FeatureBank(Path(path).resolve())
    if len(banks) < 2:
        raise ValueError('Supply at least one --bank LABEL=DIR')
    # Preserve both feature and source identities; no cross-version comparison
    # of different images or masks is allowed.
    masks = resolve_masks(mask_root, sorted(selected))
    features, images = {}, {}
    anchor_by_name = {a['name']: a for a in anchors}
    polys = diagnosis['manual_blade_roi_polygons_960x540']
    rows = []
    for name in sorted(selected):
        raw = source_path(image_root, name).read_bytes()
        source_hash = sha256(raw).hexdigest()
        with Image.open(image_root/name) as im:
            images[name] = im.convert('RGB')
        mask_bytes = masks[name].read_bytes() if name in masks else None
        mask_hash = sha256(mask_bytes).hexdigest() if mask_bytes is not None else ''
        mask = read_mask(mask_bytes, images[name].size, config['mask_threshold'])
        annotation = anchor_by_name.get(name)
        polygons = polys[str(annotation['image_id'])] if annotation else None
        for label, bank in banks.items():
            feature = bank.get(name)
            if (feature.source_sha256 != source_hash or feature.mask_sha256 != mask_hash
                    or tuple(feature.image_size) != images[name].size):
                raise ValueError(f'Image/mask identity differs: {label}, {name}')
            if mask is not None:
                pixels = np.floor(feature.keypoints).astype(np.int64)
                if not mask[pixels[:, 1], pixels[:, 0]].all():
                    raise ValueError(f'Point outside keep-mask: {label}, {name}')
            features[label, name] = feature
            row = dict(image_name=name, anchor_image_id=annotation['image_id'] if annotation else '',
                       method=label, total_keypoints=len(feature.keypoints), roi_keypoints='',
                       roi_points_per_megapixel='', roi_fraction_of_points='',
                       roi_occupied_32px_cells='', roi_cell_coverage_fraction='',
                       score_min=float(feature.scores.min()) if len(feature.scores) else None,
                       score_median=float(np.median(feature.scores)) if len(feature.scores) else None,
                       score_max=float(feature.scores.max()) if len(feature.scores) else None,
                       feature_id=feature.feature_id, feature_set_id=feature.feature_set_id)
            if polygons is not None:
                inside = roi_membership(feature.keypoints, polygons, images[name].size)
                count = int(inside.sum())
                if int(roi_mask(polygons, images[name].size).sum()) != annotation['manual_roi_pixels']:
                    raise ValueError('Rasterized ROI differs from the earlier diagnosis')
                if label == 'ALIKED grid256' and count != annotation['bank_keypoints']:
                    raise ValueError('Original ROI count differs from the earlier diagnosis')
                # Spatial coverage is reported alongside raw counts to expose
                # dense clusters. A cell counts if its center is inside the ROI.
                w, h = images[name].size
                xx, yy = np.meshgrid(np.arange(16, w, 32), np.arange(16, h, 32))
                centers = np.column_stack((xx.ravel(), yy.ravel()))
                roi_cells = centers[roi_membership(centers, polygons, images[name].size)]
                possible = set(map(tuple, (roi_cells//32).astype(int)))
                occupied = set(map(tuple, (feature.keypoints[inside]//32).astype(int))) & possible
                row.update(roi_keypoints=count,
                    roi_points_per_megapixel=count/annotation['manual_roi_pixels']*1e6,
                    roi_fraction_of_points=count/len(feature.keypoints) if len(feature.keypoints) else 0.,
                    roi_occupied_32px_cells=len(occupied),
                    roi_cell_coverage_fraction=len(occupied)/len(possible) if possible else 0.)
            rows.append(row)
    with (args.output_dir/'per_image_comparison.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(selection=selection, banks={k: dict(path=str(v.root), feature_set_id=v.set_id,
                    config=v.config['config']) for k, v in banks.items()},
                   anchor_rows=[r for r in rows if r['anchor_image_id'] != ''],
                   counts_are_detections_only=True, matching_or_3d_reliability_tested=False,
                   roi_cell_size_original_px=32, roi_cell_membership='cell center inside polygon')
    atomic_json(args.output_dir/'comparison.json', summary)
    overview = Image.new('RGB', (720*len(banks), (405+48)*len(anchors)))
    for y, anchor in enumerate(anchors):
        name = anchor['name']
        polygons = polys[str(anchor['image_id'])]
        detail = Image.new('RGB', (960*len(banks), 540+48))
        for x, label in enumerate(banks):
            overview.paste(panel(images[name], features[label, name], label, polygons), (720*x, 453*y))
            detail.paste(panel(images[name], features[label, name], label, polygons, width=960), (960*x, 0))
        detail.save(args.output_dir/f"anchor_{anchor['image_id']}.png")
    overview.save(args.output_dir/'anchors_overview.jpg', quality=95)
    for group in groups:
        canvas = Image.new('RGB', (720*len(banks), 453*len(group['views'])))
        for y, name in enumerate(group['views']):
            annotation = anchor_by_name.get(name)
            polygons = polys[str(annotation['image_id'])] if annotation else None
            for x, label in enumerate(banks):
                canvas.paste(panel(images[name], features[label, name], label, polygons), (720*x, 453*y))
        canvas.save(args.output_dir/f"neighbors_{group['image_id']}.jpg", quality=95)
    lines = ['# DeDoDe 与 ALIKED 叶片检测对照', '',
             '使用原图、相同掩膜和每图最多 4096 点；DeDoDe 不使用 ALIKED 分数阈值。',
             'ROI 沿用之前四张锚点图的人工叶片内部多边形；相邻图只展示检测分布，不套用锚点 ROI。', '',
             '| 图像 ID | 方法 | 整图点数 | 叶片 ROI 点数 | ROI 32px 格覆盖率 |',
             '|---|---|---:|---:|---:|']
    for row in summary['anchor_rows']:
        lines.append(f"| {row['anchor_image_id']} | {row['method']} | {row['total_keypoints']} | "
                     f"{row['roi_keypoints']} | {row['roi_cell_coverage_fraction']:.1%} |")
    lines += ['', '黄色为其他检测点，红色为 ROI 内点，青色为 ROI 边界。图中显示全部保存点，没有抽样。',
              'DeDoDe native top-K 使用覆盖度校正、不额外执行推理 NMS；grid256 额外施加原图网格配额。',
              '点数与格覆盖率只评价检测分布，未验证跨视图重复性、RoMa 匹配或三维重建可靠性。',
              '原始分数属于不同检测器，不能直接比较分数大小。', '',
              '[四张锚点总览](anchors_overview.jpg)', '']
    for group in groups:
        iid = group['image_id']
        lines.append(f'- ID {iid}：[锚点高清对照](anchor_{iid}.png)，[前一张 / 锚点 / 后一张](neighbors_{iid}.jpg)')
    lines += ['', '[逐图 CSV](per_image_comparison.csv) · [完整配置与统计 JSON](comparison.json)', '']
    (args.output_dir/'report.md').write_text('\n'.join(lines))
    print('\n'.join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-dir', type=Path, required=True)
    p.add_argument('--diagnosis-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--neighbors', type=int, default=1)
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--bank', action='append', default=[], metavar='LABEL=DIR')
    args = p.parse_args()
    if args.neighbors < 0:
        p.error('--neighbors must be nonnegative')
    args.baseline_dir = args.baseline_dir.resolve()
    args.diagnosis_dir = args.diagnosis_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    for root in [args.baseline_dir, args.diagnosis_dir]+[Path(s.split('=', 1)[1]).resolve() for s in args.bank]:
        if args.output_dir.is_relative_to(root) or root.is_relative_to(args.output_dir):
            p.error('comparison output must be separate from input banks and diagnosis')
    run(args)


if __name__ == '__main__':
    main()
