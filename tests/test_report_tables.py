"""Tests for the report table helpers."""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

from report_tables import compare_metric, share, t4_objects


def prompt(name, active):
    return {'object': name, 'active': active}


def label(image_id, **values):
    base = {'image_id': image_id, 'object': True, 'C': None, 'N': None, 'P': None, 'M': None, 'O2': None}
    return {**base, **values}


def test_share_ignores_unknown_values():
    assert share([True, False, None]) == 0.5


def test_compare_metric_uses_shared_images_of_selected_objects():
    prompts = {'a': prompt('cup', ['C']), 'b': prompt('dog', ['C'])}
    first = {('a', 0): label('a0', C=False), ('b', 0): label('b0', C=False)}
    second = {('a', 0): label('a0', C=True), ('b', 0): label('b0', C=False)}
    row = compare_metric(prompts, first, second, 'C', {'cup'})
    assert (row['shared_images'], row['sd15'], row['sdxl']) == (1, 0.0, 1.0)


def test_t4_objects_excludes_objects_made_on_both_gpus(tmp_path):
    path = tmp_path / 'hashes.csv'
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, ['image_id', 'sha256', 'gpu'])
        writer.writeheader()
        writer.writerows([{'image_id': 'a0', 'sha256': 'x', 'gpu': 't4'},
                          {'image_id': 'b0', 'sha256': 'x', 'gpu': 't4'},
                          {'image_id': 'b1', 'sha256': 'x', 'gpu': 'rtx3090'}])
    prompts = {'a': prompt('cup', []), 'b': prompt('dog', [])}
    labels = {('a', 0): label('a0'), ('b', 0): label('b0'), ('b', 1): label('b1')}
    assert t4_objects(prompts, labels, path) == {'cup'}
