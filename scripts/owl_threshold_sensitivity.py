"""Recompute OWLv2 count and position labels at other confidence thresholds from saved raw boxes."""
import argparse
import json
from pathlib import Path

from run_tools import (DRAWS, Bootstrap, compare_models, correct_position, filter_boxes, interference, load_run,
                       strict_value, write_csv)

THRESHOLD = 0.2
NMS_IOU = 0.5


def relabel(root, prompts, labels, thresholds):
    out = {t: {k: dict(v) for k, v in labels.items()} for t in thresholds}
    counts = {t: {} for t in thresholds}
    for key, row in labels.items():
        p = prompts[key[0]]
        if not {'N', 'P'} & set(p['active']) or row['object'] is None:
            continue
        record = json.loads((root / 'owl_records' / f"{row['image_id']}.json").read_text())
        if record['status'] != 'ok':
            continue
        for t in thresholds:
            boxes = filter_boxes(record['raw_boxes'], record['raw_scores'], t, NMS_IOU,
                                 record['width'], record['height'])
            counts[t][key] = len(boxes)
            for f in {'N', 'P'} & set(p['active']):
                if row['object'] is False:
                    value = False
                elif f == 'N':
                    value = len(boxes) == p['count']
                else:
                    value = correct_position(boxes, record['width'], p['position'])
                if t == THRESHOLD and value != row[f]:
                    raise ValueError(f"{row['image_id']} {f}: recomputed {value}, labels.csv {row[f]}")
                out[t][key][f] = value
    return out, counts


def rate(values):
    known = [v for v in values if v is not None]
    return sum(known), len(known)


def summarize(model, threshold, prompts, labels, counts):
    def active(f, size=None):
        return [r for k, r in labels.items() if f in prompts[k[0]]['active']
                and (size is None or len(prompts[k[0]]['active']) == size)]

    row = {'model': model, 'threshold': threshold}
    for f in ['N', 'P']:
        correct, images = rate(r[f] for r in active(f))
        row[f'{f}_correct'], row[f'{f}_images'], row[f'{f}_accuracy'] = correct, images, correct / images
        correct, images = rate(r[f] for r in active(f, 1))
        row[f'{f}_single_correct'], row[f'{f}_single_images'] = correct, images
        row[f'{f}_single_accuracy'] = correct / images
    correct, images = rate(strict_value(prompts[k[0]], r) for k, r in labels.items())
    row['strict_correct'], row['strict_images'], row['strict_accuracy'] = correct, images, correct / images
    signed = [c - prompts[k[0]]['count'] for k, c in counts.items()
              if 'N' in prompts[k[0]]['active'] and labels[k]['object'] is True]
    row['count_mean_signed_error'] = sum(signed) / len(signed)
    row['count_too_few_share'] = sum(s < 0 for s in signed) / len(signed)
    row['count_too_many_share'] = sum(s > 0 for s in signed) / len(signed)
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True, help='main_sd15 run directory')
    parser.add_argument('--second', required=True, help='main_sdxl run directory')
    parser.add_argument('--output', required=True)
    parser.add_argument('--thresholds', type=float, nargs='+', default=[0.2, 0.25, 0.3])
    args = parser.parse_args(argv)
    if THRESHOLD not in args.thresholds:
        raise ValueError(f'{THRESHOLD} is required to verify labels.csv')

    summary, effects, comparison, relabeled = [], [], [], {}
    for model, path in [('sd15', args.base), ('sdxl', args.second)]:
        root, prompts, pairs, labels = load_run(path)
        by_threshold, counts = relabel(root, prompts, labels, args.thresholds)
        relabeled[model] = by_threshold
        boot = Bootstrap({p['object'] for p in prompts.values()}, DRAWS, 0)
        for t in args.thresholds:
            summary.append(summarize(model, t, prompts, by_threshold[t], counts[t]))
            effects += interference(model, t, prompts, pairs, by_threshold[t], boot)
    for t in args.thresholds:
        comparison += compare_models(t, prompts, relabeled['sd15'][t], relabeled['sdxl'][t])

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / 'summary.csv', summary)
    write_csv(out / 'interference.csv', effects)
    write_csv(out / 'model_comparison.csv', comparison)
    print(f'Verified threshold {THRESHOLD} against labels.csv. Results: {out}')


if __name__ == '__main__':
    main()
