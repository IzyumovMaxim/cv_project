"""Coverage-aware descriptive metrics; no fabricated results or significance claims."""
import html
from pathlib import Path
from .evaluation import export_labels
from .prompts import FACTORS, load_prompts
from .storage import atomic_json, read_csv, records, valid_image, verify_run, write_csv


def metric(values, requested):
    known = [v for v in values if v is not None]
    correct = sum(v is True for v in known)
    return {'correct': correct, 'evaluated': len(known), 'requested': requested,
            'accuracy': correct / len(known) if known else None,
            'coverage': len(known) / requested if requested else None}


def compute_metrics(prompts, jobs, labels, pairs):
    ps = {p['prompt_id']: p for p in prompts}
    ls = {r['image_id']: r for r in labels}
    attributes, singletons = [], []
    for f in ['object'] + FACTORS:
        applicable = [j for j in jobs if f == 'object' or f in ps[j['prompt_id']]['active']]
        attributes.append({'factor': f, **metric([ls.get(j['image_id'], {}).get(f) for j in applicable], len(applicable))})
        if f != 'object':
            singleton_jobs = [j for j in applicable if ps[j['prompt_id']]['complexity'] == 1]
            singletons.append({'factor': f, **metric(
                [ls.get(j['image_id'], {}).get(f) for j in singleton_jobs], len(singleton_jobs))})
    strict_values, by_complexity = [], []
    for j in jobs:
        p, r = ps[j['prompt_id']], ls.get(j['image_id'], {})
        vals = [r.get(f) for f in ['object'] + p['active']]
        strict_values.append((p['complexity'], all(vals) if all(v is not None for v in vals) else None))
    for complexity in range(6):
        values = [v for c, v in strict_values if c == complexity]
        by_complexity.append({'complexity': complexity, **metric(values, len(values))})
    seeds = sorted({int(j['seed']) for j in jobs})
    indexed = {(r['prompt_id'], int(r['seed'])): r for r in labels}
    interference = []
    for a in FACTORS:
        for b in FACTORS:
            if a == b:
                continue
            potential = [p for p in pairs if p['added_factor'] == b and a in ps[p['base_id']]['active']]
            comparisons = []
            for pair in potential:
                for seed in seeds:
                    x = indexed.get((pair['base_id'], seed), {})
                    y = indexed.get((pair['added_id'], seed), {})
                    if x.get('object') is True and y.get('object') is True and x.get(a) is not None and y.get(a) is not None:
                        comparisons.append((x[a], y[a]))
            n = len(comparisons)
            flips = sum(x and not y for x, y in comparisons)
            base_correct = sum(x for x, _ in comparisons)
            interference.append({'checked_factor': a, 'added_factor': b, 'eligible_pairs': n,
                'potential_pairs': len(potential)*len(seeds),
                'delta_accuracy': sum(int(y)-int(x) for x, y in comparisons)/n if n else None,
                'flips': flips, 'flip_rate': flips/n if n else None,
                'base_correct_pairs': base_correct,
                'conditional_loss_rate': flips/base_correct if base_correct else None})
    return {'attributes': attributes, 'singletons': singletons,
            'strict': metric([v for _, v in strict_values], len(jobs)),
            'by_complexity': by_complexity, 'interference': interference}


def plots(out, metrics):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    attrs = metrics['attributes']
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([r['factor'] for r in attrs], [r['accuracy'] if r['accuracy'] is not None else float('nan') for r in attrs])
    ax.set(ylim=(0, 1), ylabel='Accuracy among known labels', title='Attribute accuracy (see CSV for coverage)')
    fig.tight_layout(); fig.savefig(out / 'attribute_accuracy.png', dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    rows = metrics['by_complexity']
    ax.plot([r['complexity'] for r in rows], [r['accuracy'] if r['accuracy'] is not None else float('nan') for r in rows], marker='o')
    ax.set(ylim=(0, 1), xlabel='Number of active factors (object always present)', ylabel='Strict accuracy')
    fig.tight_layout(); fig.savefig(out / 'complexity.png', dpi=160); plt.close(fig)
    matrix = np.full((5, 5), np.nan)
    for r in metrics['interference']:
        if r['delta_accuracy'] is not None:
            matrix[FACTORS.index(r['checked_factor']), FACTORS.index(r['added_factor'])] = r['delta_accuracy']
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, vmin=-1, vmax=1, cmap='RdBu')
    ax.set_xticks(range(5), FACTORS); ax.set_yticks(range(5), FACTORS)
    ax.set(xlabel='Added factor', ylabel='Checked factor', title='Accuracy after addition minus before')
    fig.colorbar(im, ax=ax); fig.tight_layout(); fig.savefig(out / 'interference.png', dpi=160); plt.close(fig)


def analyze(cfg, root):
    run = verify_run(cfg, root)
    root = Path(root)
    gens = records(root, 'generation_records')
    for g in gens.values():
        if g['status'] == 'ok' and not valid_image(root / g['path'], expected_hash=g['sha256']):
            raise ValueError('A generated image is missing or corrupt. Rerun generate and evaluate first.')
    labels = export_labels(cfg, root)
    prompts, jobs = load_prompts(root), read_csv(root / 'jobs.csv')
    result = compute_metrics(prompts, jobs, labels, read_csv(root / 'pairs.csv'))
    successful = [g for g in gens.values() if g['status'] == 'ok']
    errors = sum(g['status'] == 'error' for g in gens.values())
    qwen = records(root, 'qwen_records')
    owl = records(root, 'owl_records')
    qwen_errors = sum(r['status'] == 'error' for r in qwen.values())
    owl_errors = sum(r['status'] == 'error' for r in owl.values())
    result['run'] = {'split': cfg['dataset']['split'], 'config_hash': run['config_hash'],
        'expected_images': len(jobs), 'successful_images': len(successful),
        'generation_errors': errors, 'generation_pending': len(jobs)-len(gens),
        'generation_error_rate_of_attempted': errors/len(gens) if gens else None,
        'mean_generation_seconds': sum(g['seconds'] for g in successful)/len(successful) if successful else None,
        'qwen_errors': qwen_errors,
        'owl_errors': owl_errors,
        'qwen_error_rate_of_attempted': qwen_errors/len(qwen) if qwen else None,
        'owl_error_rate_of_attempted': owl_errors/len(owl) if owl else None,
        'qwen_unknown_answers': sum(v is None for r in qwen.values() for v in r.get('parsed', {}).values()),
        'generation_environment_ids': sorted({g['environment_id'] for g in successful if g.get('environment_id')}),
        'complete_generation': len(successful) == len(jobs),
        'complete_scoring': result['strict']['evaluated'] == len(jobs)}
    out = root / 'analysis'
    atomic_json(out / 'summary.json', result)
    for name in ['attributes', 'singletons', 'by_complexity', 'interference']:
        write_csv(out / f'{name}.csv', result[name])
    plots(out, result)
    ps = {p['prompt_id']: p for p in prompts}
    failures = []
    for r in labels:
        p = ps[r['prompt_id']]
        failed = [f for f in ['object'] + p['active'] if r.get(f) is False]
        if failed:
            failures.append({'image_id': r['image_id'], 'prompt': p['prompt'],
                             'failed_factors': '|'.join(failed), 'path': r['path']})
    selected = failures[:6]
    write_csv(out / 'failure_cases.csv', selected, ['image_id', 'prompt', 'failed_factors', 'path'])
    cards = [f'<article><p>{html.escape(r["prompt"])}</p><p>Automatic failures: {r["failed_factors"]}</p>'
             f'<img width="384" src="../{html.escape(r["path"])}"></article>' for r in selected]
    (out / 'failure_cases.html').write_text('<!doctype html><meta charset="utf-8"><title>Failure cases</title>'
        '<h1>Automatically identified failure cases</h1><p>Verify visually before using in the report. '
        f'{len(failures)} failures available; displaying {len(selected)}. No examples are fabricated.</p>' + ''.join(cards))
    if len(selected) < 3:
        print('Fewer than three scored failures available; gallery includes only observed failures.')
    print(f'Analysis written to {out}; generation complete={result["run"]["complete_generation"]}, scoring complete={result["run"]["complete_scoring"]}')
    return result
