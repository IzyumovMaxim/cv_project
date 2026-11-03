"""Check whether generator conclusions change when Qwen replaces OWLv2 for count and position."""
import argparse
from pathlib import Path

from run_tools import DRAWS, Bootstrap, compare_models, interference, load_run, strict_value, write_csv


def qwen_only(labels):
    out = {}
    for key, row in labels.items():
        new = dict(row)
        for factor in ['N', 'P']:
            if row['object'] is False and row[factor] is not None:
                new[factor] = False
            elif row['object'] is True and row[f'qwen_{factor}'] is not None:
                new[factor] = row[f'qwen_{factor}']
        out[key] = new
    return out


def accuracy(values):
    known = [v for v in values if v is not None]
    return sum(known) / len(known)


def summarize(model, setting, prompts, labels):
    row = {'model': model, 'evaluator': setting}
    for factor in ['N', 'P']:
        items = [(k, r) for k, r in labels.items() if factor in prompts[k[0]]['active']]
        row[f'{factor}_accuracy'] = accuracy(r[factor] for _, r in items)
        row[f'{factor}_single_accuracy'] = accuracy(r[factor] for k, r in items if len(prompts[k[0]]['active']) == 1)
    row['strict_accuracy'] = accuracy(strict_value(prompts[k[0]], r) for k, r in labels.items())
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True, help='main_sd15 run directory')
    parser.add_argument('--second', required=True, help='main_sdxl run directory')
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)

    summary, effects, comparison, settings = [], [], [], {}
    for model, path in [('sd15', args.base), ('sdxl', args.second)]:
        _, prompts, pairs, labels = load_run(path)
        boot = Bootstrap({p['object'] for p in prompts.values()}, DRAWS, 0)
        for setting, values in [('qwen_plus_owl', labels), ('qwen_only', qwen_only(labels))]:
            settings[(model, setting)] = values
            summary.append(summarize(model, setting, prompts, values))
            effects += interference(model, setting, prompts, pairs, values, boot)
    for setting in ['qwen_plus_owl', 'qwen_only']:
        comparison += compare_models(setting, prompts, settings[('sd15', setting)], settings[('sdxl', setting)])

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / 'summary.csv', summary)
    write_csv(out / 'interference.csv', effects)
    write_csv(out / 'model_comparison.csv', comparison)
    print(f'Results: {out}')


if __name__ == '__main__':
    main()
