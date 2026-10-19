"""Copy verified images into a new evaluator run without copying any labels."""
import hashlib
import shutil
from pathlib import Path
from .storage import (atomic_json, export_manifest, read_csv, read_json, records,
                      put_record, valid_image, verify_run)


def reuse_generation(cfg, root, source):
    root, source = Path(root).resolve(), Path(source).resolve()
    target_run = verify_run(cfg, root)
    if root == source:
        raise ValueError('Source and destination must be different run directories.')
    old = read_json(source / 'run.json')
    for section in ['dataset', 'generation']:
        if old['config'][section] != cfg[section]:
            raise ValueError(f'{section} settings differ; images cannot be reused.')
    for name, expected in target_run['data_hashes'].items():
        actual = hashlib.sha256((source / name).read_bytes()).hexdigest()
        if actual != expected or actual != old['data_hashes'][name]:
            raise ValueError(f'Prepared data differ or were changed: {name}')
    if records(root, 'generation_records') or records(root, 'qwen_records') or records(root, 'owl_records'):
        raise ValueError('Destination already contains results; reuse only into a fresh prepared run.')
    # Validate before writing, including model/environment provenance when present.
    metadata = [source / p for p in ['models/generator.json', 'scheduler.json', 'environment/generation.json']]
    metadata += list((source / 'environment' / 'sessions').glob('*.json'))
    for path in metadata:
        if path.exists():
            destination = root / path.relative_to(source)
            if destination.exists() and read_json(destination) != read_json(path):
                raise ValueError(f'Conflicting provenance: {path.name}')
    jobs = {j['image_id']: j for j in read_csv(root / 'jobs.csv')}
    selected, skipped = [], 0
    for key, row in records(source, 'generation_records').items():
        if row['status'] != 'ok':
            skipped += 1
            continue
        job = jobs.get(key)
        if not job or row['prompt_id'] != job['prompt_id'] or int(row['seed']) != int(job['seed']):
            raise ValueError(f'Generation record does not match scheduled job: {key}')
        image = (source / row['path']).resolve()
        if not image.is_relative_to(source) or not row.get('sha256'):
            raise ValueError('Invalid image path or missing image hash.')
        if not valid_image(image, [cfg['generation']['width'], cfg['generation']['height']], row['sha256']):
            skipped += 1
            continue
        selected.append((key, row, image))
    for path in metadata:
        if path.exists():
            atomic_json(root / path.relative_to(source), read_json(path))
    atomic_json(root / 'generation_import.json', {
        'source_run': str(source), 'source_config_hash': old['config_hash'],
        'source_hash': old['source_hash'], 'imported_count': len(selected),
        'skipped_count': skipped, 'note': 'Original image and generation provenance retained; no evaluation reused.'})
    for key, row, image in selected:
        relative = f'images/{key}.png'
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp = destination.with_suffix('.tmp.png')
        shutil.copyfile(image, temp)
        temp.replace(destination)
        put_record(root, 'generation_records', key, {**row, 'path': relative,
                   'imported_from': str(source), 'generation_source_hash': old['source_hash']})
    export_manifest(root)
    print(f'Reused {len(selected)} images; skipped {skipped} failed/corrupt records. Evaluations start fresh.')
