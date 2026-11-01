import importlib.util
from itertools import combinations
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[1] / 'scripts' / 'export_human_sample.py'
spec = importlib.util.spec_from_file_location('export_human_sample', path)
sample = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sample)

FACTORS = ['C', 'N', 'P', 'M', 'O2']


def dataset():
    prompts, generated = {}, []
    for obj in range(20):
        for k in range(6):
            for active in combinations(FACTORS, k):
                pid = f'{obj}-{"".join(active)}'
                prompts[pid] = {'object': str(obj), 'active': list(active)}
                generated += [{'image_id': f'{pid}_s{s}', 'prompt_id': pid, 'seed': s} for s in range(4)]
    return prompts, generated


def test_select_balances_and_is_deterministic():
    prompts, generated = dataset()
    picked = sample.select(generated, prompts, 150)
    assert picked == sample.select(list(reversed(generated)), prompts, 150)
    info = sample.summary(picked, prompts)
    assert info['complexity'] == {k: 30 for k in range(1, 6)}
    assert info['factors'] == {f: 90 for f in sorted(FACTORS)}
    assert info['objects'] == 20


def test_select_skips_object_only_prompts():
    prompts, generated = dataset()
    picked = sample.select(generated, prompts, 20)
    assert all(prompts[g['prompt_id']]['active'] for g in picked)


def test_select_refuses_too_small_pool():
    prompts, generated = dataset()
    with pytest.raises(ValueError, match='Need'):
        sample.select(generated[:4], prompts, 5)
