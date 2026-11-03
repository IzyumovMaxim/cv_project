"""Compare automatic labels with human annotations and measure inter-annotator agreement."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.proportion import proportion_confint

from run_tools import DRAWS, FACTORS, load_labels, load_prompts, mcnemar_p, read_csv, write_csv

TYPOS = {'ture': 'true', 'falsa': 'false'}


def read_annotations(path):
    with open(path, newline='', encoding='utf-8-sig') as f:
        header = f.readline()
        f.seek(0)
        return list(csv.DictReader(f, delimiter=';' if ';' in header else ','))


def clean_value(value):
    text = value.strip().lower()
    return TYPOS.get(text, text)


def clean_annotations(rows):
    clean, fixes = [], []
    for row in rows:
        active = row['active_factors'].split('|') if row['active_factors'] else []
        out = {k: v.strip() for k, v in row.items()}
        out['human_object'] = clean_value(row['human_object'])
        for f in FACTORS:
            field = f'human_{f}'
            value = clean_value(row[field])
            if f not in active:
                if value:
                    fixes.append({'image_id': row['image_id'], 'field': field, 'raw': row[field].strip(), 'clean': '',
                                  'kind': 'inactive_value_ignored'})
                value = ''
            elif row[field].strip().lower() in TYPOS:
                fixes.append({'image_id': row['image_id'], 'field': field, 'raw': row[field].strip(), 'clean': value,
                              'kind': 'typo_fixed'})
            out[field] = value
        clean.append(out)
    return clean, fixes


def kappa(first, second):
    if len(set(first) | set(second)) < 2:
        return None
    return float(cohen_kappa_score(first, second, labels=[False, True]))


def bootstrap_kappa(items, seed=0):
    groups = {}
    for group, auto, human in items:
        groups.setdefault(group, []).append((auto, human))
    names = sorted(groups)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(DRAWS):
        sample = [pair for k in rng.integers(0, len(names), len(names)) for pair in groups[names[k]]]
        value = kappa([a for a, _ in sample], [h for _, h in sample])
        if value is not None:
            draws.append(value)
    if not draws:
        return None, None
    low, high = np.percentile(draws, [2.5, 97.5])
    return float(low), float(high)


def automatic_label(row, factor, evaluator):
    if evaluator == 'qwen_plus_owl' or factor not in {'N', 'P'}:
        return row[factor]
    if row['object'] is None:
        return None
    return False if row['object'] is False else row[f'qwen_{factor}']


def agreement_items(clean, labels, prompts, factor, evaluator):
    by_image = {r['image_id']: r for r in labels.values()}
    items = []
    for row in clean:
        human = row[f'human_{factor}']
        auto = automatic_label(by_image[row['image_id']], factor, evaluator)
        if human in {'true', 'false'} and auto is not None:
            items.append((prompts[row['prompt_id']]['object'], auto, human == 'true', row['image_id']))
    return items


def agreement_row(evaluator, factor, items):
    auto = [i[1] for i in items]
    human = [i[2] for i in items]
    correct = sum(a == h for a, h in zip(auto, human))
    low, high = proportion_confint(correct, len(items), method='wilson')
    kappa_low, kappa_high = bootstrap_kappa([i[:3] for i in items])
    return {'evaluator': evaluator, 'factor': factor, 'images': len(items), 'agreed': correct,
            'accuracy': correct / len(items), 'wilson_low': low, 'wilson_high': high,
            'kappa': kappa(auto, human), 'kappa_low': kappa_low, 'kappa_high': kappa_high,
            'auto_yes_human_yes': sum(a and h for a, h in zip(auto, human)),
            'auto_yes_human_no': sum(a and not h for a, h in zip(auto, human)),
            'auto_no_human_yes': sum(h and not a for a, h in zip(auto, human)),
            'auto_no_human_no': sum(not a and not h for a, h in zip(auto, human)),
            'auto_yes_rate': sum(auto) / len(items), 'human_yes_rate': sum(human) / len(items)}


def evaluator_comparison(clean, labels, prompts):
    rows = []
    for factor in ['N', 'P']:
        owl = {i[3]: i[1] == i[2] for i in agreement_items(clean, labels, prompts, factor, 'qwen_plus_owl')}
        qwen = {i[3]: i[1] == i[2] for i in agreement_items(clean, labels, prompts, factor, 'qwen_only')}
        shared = owl.keys() & qwen.keys()
        only_owl = sum(owl[k] and not qwen[k] for k in shared)
        only_qwen = sum(qwen[k] and not owl[k] for k in shared)
        rows.append({'factor': factor, 'images': len(shared), 'owl_accuracy': sum(owl[k] for k in shared) / len(shared),
                     'qwen_accuracy': sum(qwen[k] for k in shared) / len(shared), 'only_owl_correct': only_owl,
                     'only_qwen_correct': only_qwen, 'mcnemar_p': mcnemar_p(only_owl, only_qwen)})
    return rows


def qwen_count(run, image_id):
    path = Path(run) / 'qwen_int8_records' / f'{image_id}.json'
    return json.loads(path.read_text()).get('raw', {}).get('N', {}).get('parsed_count')


def error_row(source, pairs):
    errors = [estimate - reference for estimate, reference in pairs]
    return {'comparison': source, 'images': len(errors), 'exact_share': sum(e == 0 for e in errors) / len(errors),
            'mean_signed_error': sum(errors) / len(errors), 'mean_absolute_error': sum(abs(e) for e in errors) / len(errors),
            'too_many_share': sum(e > 0 for e in errors) / len(errors),
            'too_few_share': sum(e < 0 for e in errors) / len(errors)}


def count_errors(clean, labels, prompts, run):
    by_image = {r['image_id']: r for r in labels.values()}
    generator, owl_human, owl_requested, qwen_human = [], [], [], []
    for row in clean:
        if 'N' not in row['active_factors'].split('|') or not row['notes'].isdigit() or int(row['notes']) == 0:
            continue
        human, requested = int(row['notes']), prompts[row['prompt_id']]['count']
        generator.append((human, requested))
        owl = by_image[row['image_id']]['owl_count']
        if owl is not None:
            owl_human.append((owl, human))
            owl_requested.append((owl, requested))
        qwen = qwen_count(run, row['image_id'])
        if qwen is not None:
            qwen_human.append((qwen, human))
    return [error_row('human_count_vs_requested', generator), error_row('owl_count_vs_requested', owl_requested),
            error_row('owl_count_vs_human_count', owl_human), error_row('qwen_count_vs_human_count', qwen_human)]


def inter_annotator(first_path, second_path):
    first = {r['image_id']: r for r in read_csv(first_path)}
    second = {r['image_id']: r for r in read_csv(second_path)}
    rows = []
    for factor in ['object', *FACTORS]:
        field = f'human_{factor}'
        pairs = [(clean_value(first[k][field]) == 'true', clean_value(second[k][field]) == 'true')
                 for k in first.keys() & second.keys()
                 if clean_value(first[k][field]) in {'true', 'false'} and clean_value(second[k][field]) in {'true', 'false'}]
        rows.append({'factor': factor, 'images': len(pairs), 'agreed': sum(a == b for a, b in pairs),
                     'accuracy': sum(a == b for a, b in pairs) / len(pairs),
                     'kappa': kappa([a for a, _ in pairs], [b for _, b in pairs])})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', required=True)
    parser.add_argument('--run', required=True, help='main_sdxl run directory')
    parser.add_argument('--first-annotator', required=True)
    parser.add_argument('--second-annotator', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)

    prompts, labels = load_prompts(args.run), load_labels(args.run)
    rows = read_annotations(args.annotations)
    clean, fixes = clean_annotations(rows)
    agreement = []
    for factor in ['object', *FACTORS]:
        agreement.append(agreement_row('qwen_plus_owl', factor, agreement_items(clean, labels, prompts, factor, 'qwen_plus_owl')))
    for factor in ['N', 'P']:
        agreement.append(agreement_row('qwen_only', factor, agreement_items(clean, labels, prompts, factor, 'qwen_only')))

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / 'annotations_clean.csv', clean)
    write_csv(out / 'annotation_fixes.csv', fixes)
    write_csv(out / 'agreement.csv', agreement)
    write_csv(out / 'evaluator_comparison.csv', evaluator_comparison(clean, labels, prompts))
    write_csv(out / 'count_errors.csv', count_errors(clean, labels, prompts, args.run))
    write_csv(out / 'inter_annotator.csv', inter_annotator(args.first_annotator, args.second_annotator))
    print(f'{len(clean)} images, {len(fixes)} technical fixes. Results: {out}')


if __name__ == '__main__':
    main()
