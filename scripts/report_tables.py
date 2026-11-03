"""Write the numbers of the final report, with Qwen labels for every attribute."""
import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from evaluator_agreement import kappa, qwen_count
from evaluator_dependence import qwen_only
from interaction_analysis import fit_checked
from owl_threshold_sensitivity import relabel
from run_tools import DRAWS, FACTORS, Bootstrap, ci, holm, interference, load_run, mcnemar_p, strict_value, write_csv

THRESHOLDS = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7]
METRICS = ['object', *FACTORS, 'strict']


def share(values):
    known = [v for v in values if v is not None]
    return sum(known) / len(known) if known else None


def metric_value(prompt, row, metric):
    if metric == 'strict':
        return strict_value(prompt, row)
    if metric == 'object':
        return row['object']
    return row[metric] if metric in prompt['active'] else None


def compare_metric(prompts, first, second, metric, objects=None):
    pairs = []
    for key in first.keys() & second.keys():
        prompt = prompts[key[0]]
        if objects is not None and prompt['object'] not in objects:
            continue
        x, y = metric_value(prompt, first[key], metric), metric_value(prompt, second[key], metric)
        if x is not None and y is not None:
            pairs.append((x, y))
    only_first = sum(x and not y for x, y in pairs)
    only_second = sum(y and not x for x, y in pairs)
    return {'metric': metric, 'shared_images': len(pairs),
            'sd15': sum(x for x, _ in pairs) / len(pairs), 'sdxl': sum(y for _, y in pairs) / len(pairs),
            'mcnemar_p': mcnemar_p(only_first, only_second)}


def single_accuracy(prompts, labels, metric):
    return share(r[metric] for k, r in labels.items() if prompts[k[0]]['active'] == [metric])


def accuracy_rows(prompts, first, second, objects=None):
    rows = []
    for metric in METRICS:
        row = compare_metric(prompts, first, second, metric, objects)
        if objects is None:
            row['sd15_single'] = single_accuracy(prompts, first, metric) if metric in FACTORS else ''
            row['sdxl_single'] = single_accuracy(prompts, second, metric) if metric in FACTORS else ''
        rows.append(row)
    return rows


def complexity_rows(model, prompts, labels, boot):
    rows = []
    for level in sorted({p['complexity'] for p in prompts.values()}):
        items = [(prompts[k[0]]['object'], strict_value(prompts[k[0]], r)) for k, r in labels.items()
                 if prompts[k[0]]['complexity'] == level]
        items = [item for item in items if item[1] is not None]
        value, draws = boot.stat(items)
        low, high = ci(draws)
        rows.append({'model': model, 'constraints': level, 'images': len(items),
                     'strict_accuracy': value, 'ci_low': low, 'ci_high': high})
    return rows


def regression_rows(model, prompts, labels):
    rows = []
    for checked in FACTORS:
        rows += fit_checked(model, checked, prompts, labels)[0]
    for row, p in zip(rows, holm([r['cluster_p'] for r in rows])):
        row['cluster_p_holm'] = p
    return rows


def read_human(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def threshold_rows(root, prompts, labels, human):
    by_image = {r['image_id']: k for k, r in labels.items()}
    relabeled, _ = relabel(Path(root), prompts, labels, THRESHOLDS)
    rows = []
    for factor in ['N', 'P']:
        for setting in [*THRESHOLDS, 'qwen']:
            pairs = []
            for row in human:
                if row[f'human_{factor}'] not in {'true', 'false'}:
                    continue
                key = by_image[row['image_id']]
                if setting == 'qwen':
                    value = False if labels[key]['object'] is False else labels[key][f'qwen_{factor}']
                else:
                    value = relabeled[setting][key][factor]
                if value is not None:
                    pairs.append((bool(value), row[f'human_{factor}'] == 'true'))
            auto = [a for a, _ in pairs]
            truth = [h for _, h in pairs]
            rows.append({'factor': factor, 'setting': setting, 'images': len(pairs),
                         'agreed': sum(a == h for a, h in pairs), 'kappa': kappa(auto, truth),
                         'evaluator_accepts': sum(auto), 'annotator_accepts': sum(truth)})
    return rows


def count_comparison(root, labels, human):
    by_image = {r['image_id']: r for r in labels.values()}
    counts = []
    for row in human:
        if 'N' not in row['active_factors'].split('|') or not row['notes'].isdigit() or int(row['notes']) == 0:
            continue
        owl, qwen = by_image[row['image_id']]['owl_count'], qwen_count(root, row['image_id'])
        if owl is not None and qwen is not None:
            counts.append((int(row['notes']), owl, qwen))
    only_owl = sum(o == h and q != h for h, o, q in counts)
    only_qwen = sum(q == h and o != h for h, o, q in counts)
    owl_error = np.array([abs(o - h) for h, o, _ in counts])
    qwen_error = np.array([abs(q - h) for h, _, q in counts])
    return [{'images': len(counts), 'owl_exact': sum(o == h for h, o, _ in counts),
             'qwen_exact': sum(q == h for h, _, q in counts), 'exact_mcnemar_p': mcnemar_p(only_owl, only_qwen),
             'owl_mean_absolute_error': owl_error.mean(), 'qwen_mean_absolute_error': qwen_error.mean(),
             'owl_larger_error': int((owl_error > qwen_error).sum()),
             'qwen_larger_error': int((qwen_error > owl_error).sum()),
             'absolute_error_wilcoxon_p': wilcoxon(owl_error, qwen_error).pvalue}]


def t4_objects(prompts, labels, hashes):
    gpu = {r['image_id']: r['gpu'] for r in read_human(hashes)}
    used = {}
    for key, row in labels.items():
        used.setdefault(prompts[key[0]]['object'], set()).add(gpu[row['image_id']])
    return {name for name, gpus in used.items() if gpus == {'t4'}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True, help='main_sd15 run directory')
    parser.add_argument('--second', required=True, help='main_sdxl run directory')
    parser.add_argument('--annotations', default='results/evaluator_agreement/annotations_clean.csv')
    parser.add_argument('--hashes', default='results/image_hashes/main_sdxl.csv')
    parser.add_argument('--output', default='results/report_tables')
    args = parser.parse_args(argv)

    _, prompts, pairs, owl_first = load_run(args.base)
    _, _, _, owl_second = load_run(args.second)
    first, second = qwen_only(owl_first), qwen_only(owl_second)
    boot = Bootstrap({p['object'] for p in prompts.values()}, DRAWS, 0)
    human = read_human(args.annotations)
    objects = t4_objects(prompts, owl_second, args.hashes)
    subset = [p for p in pairs if prompts[p['base_id']]['object'] in objects]
    subset_boot = Bootstrap(objects, DRAWS, 0)

    def keep(labels):
        return {k: r for k, r in labels.items() if prompts[k[0]]['object'] in objects}

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / 'accuracy.csv', accuracy_rows(prompts, first, second))
    write_csv(out / 'strict_by_constraints.csv', complexity_rows('sd15', prompts, first, boot)
              + complexity_rows('sdxl', prompts, second, boot))
    write_csv(out / 'interference.csv', interference('sd15', 'qwen', prompts, pairs, first, boot)
              + interference('sdxl', 'qwen', prompts, pairs, second, boot))
    write_csv(out / 'regression_terms.csv', regression_rows('sd15', prompts, first)
              + regression_rows('sdxl', prompts, second))
    write_csv(out / 'owl_thresholds_against_humans.csv', threshold_rows(args.second, prompts, owl_second, human))
    write_csv(out / 'count_owl_against_qwen.csv', count_comparison(args.second, owl_second, human))
    write_csv(out / 't4_objects_accuracy.csv', accuracy_rows(prompts, first, second, objects))
    write_csv(out / 't4_objects_interference.csv',
              interference('sd15', 'qwen', prompts, subset, keep(first), subset_boot)
              + interference('sdxl', 'qwen', prompts, subset, keep(second), subset_boot))
    print(f'Results: {out}')


if __name__ == '__main__':
    main()
