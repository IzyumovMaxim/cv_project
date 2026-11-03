import csv
import importlib.util
import sys
from pathlib import Path

scripts = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location('evaluator_agreement', scripts / 'evaluator_agreement.py')
agreement = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agreement)

FIELDS = ['image_id', 'prompt_id', 'active_factors', 'human_object', 'human_C', 'human_N', 'human_P', 'human_M',
          'human_O2', 'notes']


def row(**values):
    return {'image_id': 'a_s0', 'prompt_id': 'a', 'active_factors': 'C|N', 'human_object': 'true', 'human_C': 'true',
            'human_N': 'true', 'human_P': '', 'human_M': '', 'human_O2': '', 'notes': '2', **values}


def kinds(fixes):
    return {(i['field'], i['kind']) for i in fixes}


def test_reads_semicolon_files(tmp_path):
    path = tmp_path / 'annotations.csv'
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, FIELDS, delimiter=';')
        writer.writeheader()
        writer.writerow(row())
    assert agreement.read_annotations(path)[0]['human_N'] == 'true'


def test_fixes_typos_case_and_spaces():
    clean, fixes = agreement.clean_annotations([row(human_C=' Ture ', human_N='TRUE ')])
    assert clean[0]['human_C'] == 'true'
    assert clean[0]['human_N'] == 'true'
    assert kinds(fixes) == {('human_C', 'typo_fixed')}


def test_ignores_inactive_values():
    clean, fixes = agreement.clean_annotations([row(human_M='True')])
    assert clean[0]['human_M'] == ''
    assert kinds(fixes) == {('human_M', 'inactive_value_ignored')}


def test_keeps_annotator_values():
    clean, fixes = agreement.clean_annotations([row(human_object='false', notes='3', human_C='True', human_N='')])
    assert clean[0]['human_C'] == 'true'
    assert clean[0]['human_N'] == ''
    assert fixes == []


def test_kappa_handles_single_class():
    assert agreement.kappa([True, True], [True, True]) is None
    assert agreement.kappa([True, False], [True, False]) == 1.0
