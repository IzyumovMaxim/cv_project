"""Select failure analysis candidates with rule based categories and write a review gallery."""
import argparse
import hashlib
import html
import json
from pathlib import Path

from PIL import Image, ImageDraw

from run_tools import load_run, read_csv, write_csv

SALT = 'failure-candidates-v1'
PER_CATEGORY = 6
THUMBNAIL = 384
BOX_COLOR = (255, 40, 40)
TITLES = {
    'position_centered': 'OWLv2 accepts position, human rejects it',
    'owl_overcount': 'Requested count is right, OWLv2 finds too many boxes',
    'generator_overcount': 'SDXL draws too many objects',
    'material_missed': 'Human accepts material, Qwen rejects it',
    'color_missed': 'Human accepts color, Qwen rejects it',
    'object_missing': 'Human finds no main object',
    'second_object_false_yes': 'Qwen accepts second object, human rejects it',
    'color_lost_with_second_object': 'Same seed pair, color lost after adding second object',
    'position_lost_with_count': 'Same seed pair, position lost after adding count',
    'material_lost_with_second_object': 'Same seed pair, material lost after adding second object',
}


def order(image_id):
    return hashlib.sha256(f'{SALT}:{image_id}'.encode()).hexdigest()


def owl_record(root, image_id):
    path = Path(root) / 'owl_records' / f'{image_id}.json'
    return json.loads(path.read_text()) if path.exists() else None


def qwen_count(root, image_id):
    path = Path(root) / 'qwen_int8_records' / f'{image_id}.json'
    return json.loads(path.read_text()).get('raw', {}).get('N', {}).get('parsed_count')


def center_offset(record):
    centers = [(b['box'][0] + b['box'][2]) / 2 for b in record['boxes']]
    return max(abs(x / record['width'] - 0.5) for x in centers) if centers else None


def human_candidates(root, prompts, labels, annotations):
    by_image = {r['image_id']: r for r in labels.values()}
    found = {name: [] for name in ['position_centered', 'owl_overcount', 'generator_overcount', 'material_missed',
                                   'color_missed', 'object_missing', 'second_object_false_yes']}

    def add(name, row, rank, note):
        found[name].append({'category': name, 'model': 'sdxl', 'images': [row['image_id']], 'rank': rank,
                            'prompt': row['prompt'], 'added_prompt': '', 'note': note})

    for row in annotations:
        auto = by_image[row['image_id']]
        active = row['active_factors'].split('|')
        requested = prompts[row['prompt_id']]['count']
        human_count = int(row['notes']) if row['notes'].isdigit() else None
        owl = auto['owl_count']
        if 'P' in active and auto['P'] is True and row['human_P'] == 'false':
            offset = center_offset(owl_record(root, row['image_id']))
            add('position_centered', row, offset, f'largest box center offset {offset:.3f} of image width')
        if 'N' in active and row['human_N'] == 'true' and auto['N'] is False and owl is not None and owl > requested:
            add('owl_overcount', row, requested - owl, f'requested {requested}, human {human_count}, OWLv2 {owl}')
        if 'N' in active and human_count is not None and human_count > requested:
            qwen = qwen_count(root, row['image_id'])
            add('generator_overcount', row, requested - human_count,
                f'requested {requested}, human {human_count}, OWLv2 {owl}, Qwen {qwen}')
        if 'M' in active and row['human_M'] == 'true' and auto['M'] is False:
            add('material_missed', row, len(active), 'human true, Qwen false')
        if 'C' in active and row['human_C'] == 'true' and auto['C'] is False:
            add('color_missed', row, len(active), 'human true, Qwen false')
        if row['human_object'] == 'false':
            add('object_missing', row, len(active), f"human false, Qwen {auto['object']}")
        if 'O2' in active and row['human_O2'] == 'false' and auto['O2'] is True:
            add('second_object_false_yes', row, len(active), 'human false, Qwen true')
    return found


def pair_candidates(model, prompts, pairs, labels, name, checked, added):
    found = []
    for pair in pairs:
        base, new = prompts[pair['base_id']], prompts[pair['added_id']]
        if pair['added_factor'] != added or base['active'] != [checked]:
            continue
        for key, x in labels.items():
            if key[0] != pair['base_id']:
                continue
            y = labels.get((pair['added_id'], key[1]))
            if y and x['object'] is True and y['object'] is True and x[checked] is True and y[checked] is False:
                found.append({'category': name, 'model': model, 'images': [x['image_id'], y['image_id']], 'rank': 0,
                              'prompt': base['prompt'], 'added_prompt': new['prompt'],
                              'note': f'{checked} true before and false after adding {added}, automatic labels'})
    return found


def pick(items):
    return sorted(items, key=lambda c: (c['rank'], order(c['images'][0])))[:PER_CATEGORY]


def thumbnail(root, image_id, out, boxes):
    target = out / 'thumbs' / f'{Path(root).name}_{image_id}.jpg'.replace(' ', '_')
    image = Image.open(Path(root) / 'images' / f'{image_id}.png').convert('RGB')
    if boxes:
        record = owl_record(root, image_id)
        draw = ImageDraw.Draw(image)
        for box in (record['boxes'] if record else []):
            draw.rectangle(box['box'], outline=BOX_COLOR, width=max(2, image.width // 200))
    image.thumbnail((THUMBNAIL, THUMBNAIL))
    image.save(target, quality=85)
    return target


def write_sheet(out, category, cards):
    width = sum(len(c['paths']) for c in cards[:1]) * THUMBNAIL
    rows = (len(cards) + 2) // 3
    sheet = Image.new('RGB', (width * 3, rows * (THUMBNAIL + 24)), 'white')
    draw = ImageDraw.Draw(sheet)
    for index, card in enumerate(cards):
        x, y = (index % 3) * width, (index // 3) * (THUMBNAIL + 24)
        draw.text((x + 4, y + 6), f"{card['id']} {card['model']}", fill='black')
        for k, path in enumerate(card['paths']):
            sheet.paste(Image.open(path), (x + k * THUMBNAIL, y + 24))
    sheet.save(out / 'sheets' / f'{category}.jpg', quality=85)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True, help='main_sd15 run directory')
    parser.add_argument('--second', required=True, help='main_sdxl run directory')
    parser.add_argument('--annotations', required=True, help='annotations_clean.csv')
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)

    out = Path(args.output)
    (out / 'thumbs').mkdir(parents=True, exist_ok=True)
    (out / 'sheets').mkdir(exist_ok=True)
    roots = {'sd15': args.base, 'sdxl': args.second}
    runs = {model: load_run(path) for model, path in roots.items()}
    _, prompts, _, labels = runs['sdxl']
    groups = human_candidates(args.second, prompts, labels, read_csv(args.annotations))
    for name, checked, added in [('color_lost_with_second_object', 'C', 'O2'), ('position_lost_with_count', 'P', 'N'),
                                 ('material_lost_with_second_object', 'M', 'O2')]:
        groups[name] = []
        for model in roots:
            _, model_prompts, pairs, model_labels = runs[model]
            groups[name] += pick(pair_candidates(model, model_prompts, pairs, model_labels, name, checked, added))

    rows, sections = [], []
    for category, items in groups.items():
        chosen = pick(items) if not category.endswith(('second_object', 'with_count')) else items
        cards = []
        for number, item in enumerate(chosen, 1):
            uid = f'{category}_{number}'
            boxes = category in {'position_centered', 'owl_overcount', 'generator_overcount', 'position_lost_with_count'}
            paths = [thumbnail(roots[item['model']], image_id, out, boxes) for image_id in item['images']]
            cards.append({**item, 'id': uid, 'paths': paths})
            rows.append({'id': uid, 'category': category, 'model': item['model'], 'images': '|'.join(item['images']),
                         'prompt': item['prompt'], 'added_prompt': item['added_prompt'], 'note': item['note']})
        if not cards:
            continue
        write_sheet(out, category, cards)
        body = ''.join(
            f"<article><h3>{c['id']} {c['model']}</h3><p>{html.escape(c['prompt'])}</p>"
            + (f"<p>{html.escape(c['added_prompt'])}</p>" if c['added_prompt'] else '')
            + ''.join(f'<img loading="lazy" src="{html.escape(str(p.relative_to(out)))}">' for p in c['paths'])
            + f"<p>{html.escape(c['note'])}</p></article>" for c in cards)
        sections.append(f'<section><h2>{html.escape(TITLES[category])}</h2>{body}</section>')
    write_csv(out / 'candidates.csv', rows)
    (out / 'gallery.html').write_text(
        '<!doctype html><meta charset="utf-8"><title>Failure candidates</title><style>'
        'body{font-family:sans-serif;margin:24px}article{display:inline-block;vertical-align:top;margin:8px;width:400px}'
        'img{width:192px;margin:2px}</style><h1>Failure candidates</h1>' + ''.join(sections), encoding='utf-8')
    print(f'{len(rows)} candidates. Results: {out}')


if __name__ == '__main__':
    main()
