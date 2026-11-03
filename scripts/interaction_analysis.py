"""Estimate non-additivity with additive deltas, joint success ratios and mixed logistic regressions."""
import argparse
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import scipy.sparse as sparse
import statsmodels.api as sm
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

from run_tools import DRAWS, FACTORS, Bootstrap, ci, compute_metrics, holm, load_run, read_csv, write_csv

Z = 1.96


def additive_tables(model, root, prompts, pairs, labels):
    jobs = read_csv(Path(root) / 'jobs.csv')
    metrics = compute_metrics(list(prompts.values()), jobs, list(labels.values()), pairs)
    detail = [{'model': model, **r} for r in metrics['nonadditivity']]
    summary = [{'model': model, **r} for r in metrics['nonadditivity_summary']]
    return detail, summary


def joint_ratios(model, prompts, labels, boot):
    rows = []
    def collect(active, check):
        items = []
        for key, row in labels.items():
            p = prompts[key[0]]
            if set(p['active']) == active and row['object'] is True:
                values = [row[f] for f in check]
                if None not in values:
                    items.append((p['object'], all(values)))
        return boot.stat(items)
    for a, b in combinations(FACTORS, 2):
        pa, pa_draws = collect({a}, [a])
        pb, pb_draws = collect({b}, [b])
        pab, pab_draws = collect({a, b}, [a, b])
        with np.errstate(invalid='ignore', divide='ignore'):
            low, high = ci(pab_draws / (pa_draws * pb_draws))
        rows.append({'model': model, 'first': a, 'second': b, 'first_alone': pa, 'second_alone': pb, 'both': pab,
                     'expected_if_independent': pa * pb, 'ratio': pab / (pa * pb), 'ci_low': low, 'ci_high': high})
    return rows


def design(checked, prompts, labels):
    others = [f for f in FACTORS if f != checked]
    pairs = list(combinations(others, 2))
    names = ['intercept', *others, *[f'{x}:{y}' for x, y in pairs]]
    y, x, objects, seeds = [], [], [], []
    for key, row in labels.items():
        p = prompts[key[0]]
        if checked not in p['active'] or row['object'] is not True or row[checked] is None:
            continue
        active = {f: int(f in p['active']) for f in others}
        y.append(int(row[checked]))
        x.append([1, *[active[f] for f in others], *[active[a] * active[b] for a, b in pairs]])
        objects.append(p['object'])
        seeds.append(row['seed'])
    return np.array(y), np.array(x, dtype=float), names, np.array(objects), np.array(seeds)


def dummies(values):
    levels, codes = np.unique(values, return_inverse=True)
    return sparse.csr_matrix((np.ones(len(codes)), (np.arange(len(codes)), codes)), shape=(len(codes), len(levels)))


def fit_checked(model, checked, prompts, labels):
    y, x, names, objects, seeds = design(checked, prompts, labels)
    vc = sparse.hstack([dummies(objects), dummies(seeds)]).tocsr()
    ident = np.array([0] * len(np.unique(objects)) + [1] * len(np.unique(seeds)))
    mixed = BinomialBayesMixedGLM(y, x, vc, ident, vcp_p=1, fe_p=5, fep_names=names, vcp_names=['object', 'seed'])
    variational = mixed.fit_vb()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        posterior_mode = mixed.fit_map()
    groups = np.unique(objects, return_inverse=True)[1]
    robust = sm.GLM(y, x, family=sm.families.Binomial()).fit(cov_type='cluster', cov_kwds={'groups': groups})
    rows = []
    for k, name in enumerate(names):
        if name == 'intercept':
            continue
        rows.append({'model': model, 'checked': checked, 'term': name, 'kind': 'interaction' if ':' in name else 'added',
                     'images': len(y), 'log_odds': variational.fe_mean[k], 'posterior_sd': variational.fe_sd[k],
                     'low': variational.fe_mean[k] - Z * variational.fe_sd[k],
                     'high': variational.fe_mean[k] + Z * variational.fe_sd[k],
                     'odds_ratio': float(np.exp(variational.fe_mean[k])), 'map_log_odds': posterior_mode.fe_mean[k],
                     'cluster_log_odds': robust.params[k], 'cluster_se': robust.bse[k], 'cluster_p': robust.pvalues[k]})
    fit = {'model': model, 'checked': checked, 'images': len(y), 'success_rate': float(y.mean()),
           'object_sd': float(np.exp(variational.vcp_mean[0])), 'seed_sd': float(np.exp(variational.vcp_mean[1])),
           'map_converged': not any('did not converge' in str(w.message) for w in caught)}
    return rows, fit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True, help='main_sd15 run directory')
    parser.add_argument('--second', required=True, help='main_sdxl run directory')
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)

    detail, summary, ratios, terms, fits = [], [], [], [], []
    for model, path in [('sd15', args.base), ('sdxl', args.second)]:
        root, prompts, pairs, labels = load_run(path)
        boot = Bootstrap({p['object'] for p in prompts.values()}, DRAWS, 0)
        table, overview = additive_tables(model, root, prompts, pairs, labels)
        detail += table
        summary += overview
        ratios += joint_ratios(model, prompts, labels, boot)
        model_terms = []
        for checked in FACTORS:
            rows, fit = fit_checked(model, checked, prompts, labels)
            model_terms += rows
            fits.append(fit)
        for r, p in zip(model_terms, holm([r['cluster_p'] for r in model_terms])):
            r['cluster_p_holm'] = p
        terms += model_terms

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / 'additive_interactions.csv', detail)
    write_csv(out / 'additive_summary.csv', summary)
    write_csv(out / 'joint_ratios.csv', ratios)
    write_csv(out / 'regression_terms.csv', terms)
    write_csv(out / 'regression_fits.csv', fits)
    print(f'Results: {out}')


if __name__ == '__main__':
    main()
