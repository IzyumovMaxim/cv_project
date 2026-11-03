import csv
import hashlib
import importlib.util
from pathlib import Path

path = Path(__file__).resolve().parents[1] / 'scripts' / 'verify_images.py'
spec = importlib.util.spec_from_file_location('verify_images', path)
verify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify)


def row(image_id, data, gpu='t4'):
    return {'image_id': image_id, 'sha256': hashlib.sha256(data).hexdigest() if data else '', 'gpu': gpu}


def test_compare_reports_every_outcome(tmp_path):
    (tmp_path / 'images').mkdir()
    (tmp_path / 'images' / 'same.png').write_bytes(b'a')
    (tmp_path / 'images' / 'changed.png').write_bytes(b'b')
    rows = [row('same', b'a'), row('changed', b'c'), row('absent', b'd'), row('filtered', b'')]
    outcome, different = verify.compare(tmp_path, rows)
    assert outcome == {'matched_t4': 1, 'different_t4': 1, 'missing': 1, 'filtered_in_original_run': 1}
    assert different == ['changed']


def test_read_hashes(tmp_path):
    target = tmp_path / 'hashes.csv'
    with open(target, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, ['image_id', 'sha256', 'gpu'])
        writer.writeheader()
        writer.writerow(row('same', b'a'))
    assert verify.read_hashes(target)[0]['image_id'] == 'same'
