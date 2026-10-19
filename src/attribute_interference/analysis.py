"""Coverage-aware metrics, McNemar tests and object-cluster bootstrap CIs; no fabricated results."""
import html
import math
from itertools import combinations
from pathlib import Path
import numpy as np
from .evaluation import export_labels
from .prompts import FACTORS, load_prompts
from .storage import atomic_json, read_csv, records, valid_image, verify_run, write_csv


BOOTSTRAP = 2000
BOOTSTRAP_SEED = 0


def mcnemar_p(b, c):
    n = b + c
    if not n:
        return None
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def holm(ps):
    order = sorted((p, i) for i, p in enumerate(ps) if p is not None)
    out, running = [None] * len(ps), 0.0
    for rank, (p, i) in enumerate(order):
        running = max(running, min(1.0, (len(order) - rank) * p))
        out[i] = running
    return out


class Bootstrap:
    def __init__(self, objects, draws, seed):
        self.index = {o: k for k, o in enumerate(sorted(objects))}
        k = len(self.index)
        rng = np.random.default_rng(seed)
        self.weights = rng.multinomial(k, [1 / k] * k, size=draws) if k else np.zeros((draws, 0))

    def stat(self, items):
        num, den = np.zeros(len(self.index)), np.zeros(len(self.index))
        for obj, value in items:
            num[self.index[obj]] += value
            den[self.index[obj]] += 1
        point = float(num.sum() / den.sum()) if den.sum() else None
        with np.errstate(invalid='ignore', divide='ignore'):
            draws = self.weights @ num / (self.weights @ den)
        return point, draws


def ci(draws):
    if draws is None or np.all(np.isnan(draws)):
        return None, None
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return float(lo), float(hi)


def nonadditivity(prompts, labels, boot):
    ps = {p['prompt_id']: p for p in prompts}
    cells = {(ps[r['prompt_id']]['object'], frozenset(ps[r['prompt_id']]['active']), int(r['seed'])): r
             for r in labels}

    def diffs(a, added):
        out = []
        for (obj, active, seed), x in cells.items():
            if active != {a}:
                continue
            y = cells.get((obj, active | added, seed), {})
            if x.get('object') is True and y.get('object') is True and x.get(a) is not None and y.get(a) is not None:
                out.append((obj, int(y[a]) - int(x[a])))
        return out

    rows, summary = [], []
    for a in FACTORS:
        others = [b for b in FACTORS if b != a]
        single = {b: boot.stat(diffs(a, {b})) for b in others}
        for k in range(2, len(others) + 1):
            group = []
            for added in combinations(others, k):
                observed, observed_draws = boot.stat(diffs(a, set(added)))
                if observed is None or any(single[b][0] is None for b in added):
                    continue
                predicted = sum(single[b][0] for b in added)
                draws = observed_draws - sum(single[b][1] for b in added)
                lo, hi = ci(draws)
                group.append(draws)
                rows.append({'checked_factor': a, 'added_factors': '|'.join(added), 'k': k,
                             'observed_delta': observed, 'additive_prediction': predicted,
                             'interaction': observed - predicted, 'ci_low': lo, 'ci_high': hi})
            if group:
                points = [r['interaction'] for r in rows if r['checked_factor'] == a and r['k'] == k]
                with np.errstate(invalid='ignore'):
                    lo, hi = ci(np.nanmean(np.vstack(group), axis=0))
                summary.append({'checked_factor': a, 'k': k, 'sets': len(points),
                                'mean_interaction': sum(points) / len(points), 'ci_low': lo, 'ci_high': hi})
    return rows, summary


def metric(values, requested):
    known = [v for v in values if v is not None]
    correct = sum(v is True for v in known)
    return {'correct': correct, 'evaluated': len(known), 'requested': requested,
            'accuracy': correct / len(known) if known else None,
            'coverage': len(known) / requested if requested else None}


def compute_metrics(prompts, jobs, labels, pairs, draws=BOOTSTRAP, boot_seed=BOOTSTRAP_SEED):
    ps = {p['prompt_id']: p for p in prompts}
    boot = Bootstrap({p['object'] for p in prompts}, draws, boot_seed)
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
        strict_values.append((p['complexity'], p['object'], all(vals) if all(v is not None for v in vals) else None))
    for complexity in range(6):
        values = [(o, v) for c, o, v in strict_values if c == complexity]
        lo, hi = ci(boot.stat([(o, v) for o, v in values if v is not None])[1])
        by_complexity.append({'complexity': complexity, **metric([v for _, v in values], len(values)),
                              'ci_low': lo, 'ci_high': hi})
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
                        comparisons.append((ps[pair['base_id']]['object'], x[a], y[a]))
            n = len(comparisons)
            flips = sum(x and not y for _, x, y in comparisons)
            gains = sum(y and not x for _, x, y in comparisons)
            base_correct = sum(x for _, x, _ in comparisons)
            delta_lo, delta_hi = ci(boot.stat([(o, int(y) - int(x)) for o, x, y in comparisons])[1])
            flip_lo, flip_hi = ci(boot.stat([(o, x and not y) for o, x, y in comparisons])[1])
            interference.append({'checked_factor': a, 'added_factor': b, 'eligible_pairs': n,
                'potential_pairs': len(potential)*len(seeds),
                'delta_accuracy': sum(int(y)-int(x) for _, x, y in comparisons)/n if n else None,
                'delta_ci_low': delta_lo, 'delta_ci_high': delta_hi,
                'flips': flips, 'gains': gains, 'flip_rate': flips/n if n else None,
                'flip_ci_low': flip_lo, 'flip_ci_high': flip_hi,
                'base_correct_pairs': base_correct,
                'conditional_loss_rate': flips/base_correct if base_correct else None,
                'mcnemar_p': mcnemar_p(flips, gains)})
    for r, p in zip(interference, holm([r['mcnemar_p'] for r in interference])):
        r['mcnemar_p_holm'] = p
    rows, summary = nonadditivity(prompts, labels, boot)
    return {'attributes': attributes, 'singletons': singletons,
            'strict': metric([v for _, _, v in strict_values], len(jobs)),
            'by_complexity': by_complexity, 'interference': interference,
            'nonadditivity': rows, 'nonadditivity_summary': summary,
            'bootstrap': {'unit': 'object', 'draws': draws, 'seed': boot_seed, 'objects': len(boot.index)}}


def nan(v):
    return float('nan') if v is None else v


def errors(rows, value, low, high):
    points = np.array([nan(r[value]) for r in rows])
    return points, np.vstack([points - np.array([nan(r[low]) for r in rows]),
                              np.array([nan(r[high]) for r in rows]) - points])


def plots(out, metrics):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    attrs = metrics['attributes']
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([r['factor'] for r in attrs], [nan(r['accuracy']) for r in attrs], color='#4a6fa5')
    ax.set(ylim=(0, 1), ylabel='Accuracy among known labels', title='Attribute accuracy (see CSV for coverage)')
    fig.tight_layout(); fig.savefig(out / 'attribute_accuracy.png', dpi=160); plt.close(fig)

    rows = metrics['by_complexity']
    points, err = errors(rows, 'accuracy', 'ci_low', 'ci_high')
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar([r['complexity'] for r in rows], points, yerr=err, marker='o', capsize=3, color='#4a6fa5')
    ax.set(ylim=(0, 1), xlabel='Number of active factors (object always present)', ylabel='Strict accuracy',
           title='Strict accuracy, 95% object-bootstrap CI')
    fig.tight_layout(); fig.savefig(out / 'complexity.png', dpi=160); plt.close(fig)

    matrix = np.full((5, 5), np.nan)
    fig, ax = plt.subplots(figsize=(7, 6))
    for r in metrics['interference']:
        if r['delta_accuracy'] is None:
            continue
        i, j = FACTORS.index(r['checked_factor']), FACTORS.index(r['added_factor'])
        matrix[i, j] = r['delta_accuracy']
        q = r['mcnemar_p_holm']
        stars = '' if q is None else '***' if q < .001 else '**' if q < .01 else '*' if q < .05 else ''
        ax.text(j, i, f"{r['delta_accuracy']:+.2f}{stars}\nn={r['eligible_pairs']}", ha='center', va='center',
                fontsize=8, color='white' if abs(r['delta_accuracy']) > .5 else 'black')
    im = ax.imshow(matrix, vmin=-1, vmax=1, cmap='RdBu')
    ax.set_xticks(range(5), FACTORS); ax.set_yticks(range(5), FACTORS)
    ax.set(xlabel='Added factor', ylabel='Checked factor',
           title='Accuracy after addition minus before\n* Holm-adjusted McNemar p<.05, ** <.01, *** <.001')
    fig.colorbar(im, ax=ax); fig.tight_layout(); fig.savefig(out / 'interference.png', dpi=160); plt.close(fig)

    rows = [r for r in metrics['interference'] if r['delta_accuracy'] is not None]
    if rows:
        rows = sorted(rows, key=lambda r: r['delta_accuracy'])
        points, err = errors(rows, 'delta_accuracy', 'delta_ci_low', 'delta_ci_high')
        fig, ax = plt.subplots(figsize=(7, 0.3 * len(rows) + 1.5))
        ax.errorbar(points, range(len(rows)), xerr=err, fmt='o', capsize=3, color='#4a6fa5')
        ax.axvline(0, color='#888888', lw=1)
        ax.set_yticks(range(len(rows)), [f"{r['checked_factor']} + {r['added_factor']}" for r in rows])
        ax.set(xlabel='Delta accuracy of checked factor', ylabel='Checked + added factor',
               title='Interference, 95% object-bootstrap CI')
        fig.tight_layout(); fig.savefig(out / 'interference_forest.png', dpi=160); plt.close(fig)

    summary = metrics['nonadditivity_summary']
    if summary:
        fig, axes = plt.subplots(1, 5, figsize=(14, 3.5), sharey=True)
        for ax, f in zip(axes, FACTORS):
            rows = [r for r in summary if r['checked_factor'] == f]
            if rows:
                points, err = errors(rows, 'mean_interaction', 'ci_low', 'ci_high')
                ax.errorbar([r['k'] for r in rows], points, yerr=err, marker='o', capsize=3, color='#4a6fa5')
            ax.axhline(0, color='#888888', lw=1)
            ax.set(title=f'Checked {f}', xlabel='Added factors k', xticks=[2, 3, 4])
        axes[0].set_ylabel('Observed minus additive Delta')
        fig.suptitle('Non-additivity: mean interaction over sets of size k, 95% object-bootstrap CI')
        fig.tight_layout(); fig.savefig(out / 'nonadditivity.png', dpi=160); plt.close(fig)


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
    for name in ['attributes', 'singletons', 'by_complexity', 'interference', 'nonadditivity', 'nonadditivity_summary']:
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
