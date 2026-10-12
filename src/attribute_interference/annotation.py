"""Development-only human review export and evaluator freeze."""
import hashlib
import html
from pathlib import Path
from .evaluation import evaluator_signature, export_labels
from .prompts import FACTORS, load_prompts
from .storage import atomic_json, digest, read_csv, read_json, records, source_hash, valid_image, verify_run, write_csv


def select_review(candidates, prompts, count):
    """Deterministic greedy coverage, then balance complexity and object counts."""
    selected, covered, complexity_counts, object_counts = [], set(), {}, {}
    candidates = sorted(candidates, key=lambda r: digest(r['image_id']))
    while candidates and len(selected) < count:
        def score(r):
            p = prompts[r['prompt_id']]
            tags = {f'factor:{f}' for f in p['active']} | {f"complexity:{p['complexity']}", f"object:{p['object']}"}
            return (len(tags - covered), -complexity_counts.get(p['complexity'], 0), -object_counts.get(p['object'], 0))
        best = max(candidates, key=score)
        candidates.remove(best)
        selected.append(best)
        p = prompts[best['prompt_id']]
        covered |= {f'factor:{f}' for f in p['active']} | {f"complexity:{p['complexity']}", f"object:{p['object']}"}
        complexity_counts[p['complexity']] = complexity_counts.get(p['complexity'], 0) + 1
        object_counts[p['object']] = object_counts.get(p['object'], 0) + 1
    return selected


def export_annotations(cfg, root, count=24):
    verify_run(cfg, root)
    root = Path(root)
    out = root / 'review'
    if (out / 'annotations.csv').exists():
        raise ValueError('Review already exists; preserving human edits. Use the existing annotations.csv.')
    prompts = {p['prompt_id']: p for p in load_prompts(root)}
    generated = [g for g in records(root, 'generation_records').values() if g['status'] == 'ok']
    if len(generated) < count:
        raise ValueError(f'Need at least {count} successful images, found {len(generated)}')
    selected = select_review(generated, prompts, count)
    rows, cards = [], []
    for gen in selected:
        p = prompts[gen['prompt_id']]
        rows.append({'image_id': gen['image_id'], 'prompt_id': p['prompt_id'], 'prompt': p['prompt'],
                     'active_factors': p['active_factors'], 'reviewer': '',
                     **{f'human_{f}': '' for f in ['object'] + FACTORS}, 'notes': ''})
        cards.append(f'<article><h2>{html.escape(gen["image_id"])}</h2><p>{html.escape(p["prompt"])}</p>'
                     f'<img width="384" src="../{html.escape(gen["path"])}"></article>')
    write_csv(out / 'annotations.csv', rows)
    atomic_json(out / 'selection.json', {g['image_id']: g['sha256'] for g in selected})
    (out / 'gallery.html').write_text('<!doctype html><meta charset="utf-8"><title>Blind development review</title>'
        '<h1>Development review</h1><p>Fill annotations.csv with true, false or unknown. Leave inactive factors blank. '
        'If the main object is absent, mark all active factors false. Automatic labels are hidden.</p>' + ''.join(cards))
    print(f'Exported {count} images to {out}. Open gallery.html and fill annotations.csv.')


def human_value(value):
    return {'true': True, 'false': False, 'unknown': None}[value.strip().lower()]


def freeze_evaluator(cfg, root):
    verify_run(cfg, root)
    if cfg['dataset']['split'] != 'pilot':
        raise ValueError('Freeze only from the development pilot, never from main or smoke.')
    root = Path(root)
    annotations = root / 'review' / 'annotations.csv'
    rows = read_csv(annotations)
    selection = read_json(root / 'review' / 'selection.json')
    if len(rows) != 24 or len({r['image_id'] for r in rows}) != 24 or set(selection) != {r['image_id'] for r in rows}:
        raise ValueError('Exactly the 24 exported development images must be reviewed.')
    generated = records(root, 'generation_records')
    prompts = {p['prompt_id']: p for p in load_prompts(root)}
    labels = {r['image_id']: r for r in export_labels(cfg, root)}
    agreement = {f: {'correct': 0, 'total': 0, 'reviewed': 0} for f in ['object'] + FACTORS}
    for r in rows:
        key = r['image_id']
        if generated[key]['sha256'] != selection[key]:
            raise ValueError('An annotated image changed; repeat development review in a new run.')
        if not valid_image(root / generated[key]['path'], expected_hash=selection[key]):
            raise ValueError('A reviewed image is corrupt or missing. Restore it before freezing.')
        p = prompts[generated[key]['prompt_id']]
        if not r['reviewer'].strip():
            raise ValueError(f'Missing reviewer for {key}')
        for f in ['object'] + p['active']:
            try:
                value = human_value(r[f'human_{f}'])
            except KeyError as exc:
                raise ValueError(f'{key}: human_{f} must be true, false or unknown') from exc
            if r['human_object'].lower().strip() == 'false' and value is not False:
                raise ValueError(f'{key}: absent object requires every active attribute to be false')
            auto = labels[key][f]
            agreement[f]['reviewed'] += 1
            if value is not None and auto is not None:
                agreement[f]['total'] += 1
                agreement[f]['correct'] += int(auto == value)
    models = {role: read_json(root / 'models' / f'{role}.json') for role in ['qwen', 'owl']}
    environments = {role: read_json(root / 'environment' / f'{role}.json') for role in ['qwen', 'owl']}
    frozen = {'evaluator_signature': evaluator_signature(cfg), 'source_hash': source_hash(),
              'pilot_config_hash': digest(cfg), 'models': models, 'review_count': len(rows),
              'environments': environments,
              'annotations_sha256': hashlib.sha256(annotations.read_bytes()).hexdigest(),
              'agreement': agreement,
              'note': 'Development diagnostic only; no independent evaluator test claim.'}
    if any(a['total'] == 0 for a in agreement.values()):
        raise ValueError('Every factor needs at least one known human/automatic comparison before freezing.')
    path = root / 'evaluator_freeze.json'
    if path.exists() and read_json(path) != frozen:
        raise ValueError('Freeze is immutable. Use a new pilot run for revised decisions.')
    atomic_json(path, frozen)
    print(f'Frozen evaluator: {path}. Review agreement fields before using it on main.')
