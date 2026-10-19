"""Synthetic fixtures test software only; they are never research results."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import types

import pytest
from PIL import Image

from attribute_interference.analysis import compute_metrics, analyze, holm, mcnemar_p
from attribute_interference.annotation import export_annotations, freeze_evaluator
from attribute_interference.evaluation import (correct_position, ensure_freeze, export_labels,
    filter_boxes, fused_labels, parse_count, parse_yes_no, questions)
from attribute_interference.generation import generate, pending_jobs
from attribute_interference.prompts import build, load_prompts, prepare
from attribute_interference.storage import (atomic_json, load_config, put_record, read_csv,
    read_json, valid_image, verify_run)

BASE = Path(__file__).resolve().parents[1]


def config(name='main_sd15'):
    return load_config(BASE / 'configs' / f'{name}.yaml')


@pytest.mark.parametrize('name,expected_prompts,expected_jobs,expected_pairs', [
    ('main_sd15',576,2304,1408), ('main_sdxl',576,2304,1408),
    ('pilot_sdxl',80,160,192), ('smoke',8,8,7), ('smoke_sdxl',8,8,7)])
def test_dataset_sizes_and_unique_jobs(tmp_path, name, expected_prompts, expected_jobs, expected_pairs):
    run = prepare(config(name), tmp_path / name)
    assert (run['prompt_count'], run['job_count'], run['pair_count']) == (expected_prompts, expected_jobs, expected_pairs)
    jobs = read_csv(tmp_path / name / 'jobs.csv')
    assert len({j['image_id'] for j in jobs}) == expected_jobs


def test_pairs_and_assignments():
    rows, pairs = build(config())
    index = {r['prompt_id']: r for r in rows}
    assert len({r['prompt'] for r in rows}) == 576
    for pair in pairs:
        a,b = index[pair['base_id']], index[pair['added_id']]
        active_a = set(filter(None, a['active_factors'].split('|')))
        active_b = set(filter(None, b['active_factors'].split('|')))
        assert active_b - active_a == {pair['added_factor']}
        assert active_a <= active_b
        assert a['object'] == b['object']
        for field in ['color','count','position','material','second_object','second_color']:
            assert a[field] == b[field]
    for r in rows:
        assert r['color'] != r['second_color']
        if r['object'] in ['dog','cat','bird','horse']:
            assert r['material'] == '' and 'M' not in r['active_factors']
    assert build(config()) == (rows, pairs)
    smoke_objects = {r['object'] for r in build(config('smoke'))[0]}
    assert not smoke_objects & {r['object'] for r in rows + build(config('pilot_sdxl'))[0]}


def test_pluralization_and_questions(tmp_path):
    prepare(config(), tmp_path)
    rows = load_prompts(tmp_path)
    assert any('three boxes' in r['prompt'] or 'four boxes' in r['prompt'] or 'two boxes' in r['prompt'] for r in rows)
    for r in rows:
        assert set(questions(r)) == {'object', *r['active']}


@pytest.mark.parametrize('answer,expected', [('Yes.',True),(' no! ',False),('Yes, but no.',None),('unknown',None),('',None)])
def test_yes_no(answer, expected):
    assert parse_yes_no(answer) is expected


@pytest.mark.parametrize('answer,expected', [('three',3),(' 4. ',4),('zero',0),('3 or 4',None),('There are 3',None),('-1',None)])
def test_count(answer, expected):
    assert parse_count(answer) == expected


def test_detector_filter_and_boundary():
    boxes = [[0,0,10,10],[1,1,9,9],[60,0,70,10],[0,0,0,5]]
    kept = filter_boxes(boxes, [.9,.8,.1,.9], .2, .5, 100, 100)
    assert len(kept) == 1
    assert correct_position(kept, 100, 'left')
    assert not correct_position([], 100, 'left')
    center = [{'box':[40,0,60,10]}]
    assert not correct_position(center,100,'left')
    assert not correct_position(center,100,'right')
    assert not correct_position(kept+[{'box':[60,0,70,10]}],100,'left')


def test_missing_object_and_unknown_labels():
    p = {'active':['C','N','P'], 'count':2, 'position':'left'}
    q = {'status':'ok','parsed':{'object':False,'C':True}}
    assert fused_labels(p,q,None) == {'object':False,'C':False,'N':False,'P':False}
    assert all(v is None for v in fused_labels(p,None,None).values())
    q['parsed']['object'] = True
    assert fused_labels(p,q,None) == {'object':True,'C':True,'N':None,'P':None}
    assert fused_labels(p,q,{'status':'ok','boxes':[],'width':100})['N'] is False


def test_coverage_and_matched_pairs():
    ps = [{'prompt_id':'a','object':'cup','active':['C'],'complexity':1},
          {'prompt_id':'b','object':'cup','active':['C','N'],'complexity':2}]
    jobs = [{'image_id':f'{p}_s{s}','prompt_id':p,'seed':s} for p in ['a','b'] for s in [0,1]]
    labels = [
        {'image_id':'a_s0','prompt_id':'a','seed':0,'object':True,'C':True},
        {'image_id':'b_s0','prompt_id':'b','seed':0,'object':True,'C':False,'N':True},
        {'image_id':'a_s1','prompt_id':'a','seed':1,'object':False,'C':False},
        {'image_id':'b_s1','prompt_id':'b','seed':1,'object':True,'C':True,'N':None}]
    result = compute_metrics(ps,jobs,labels,[{'base_id':'a','added_id':'b','added_factor':'N'}])
    assert result['strict'] == {'correct':1,'evaluated':3,'requested':4,'accuracy':1/3,'coverage':.75}
    n = next(r for r in result['attributes'] if r['factor']=='N')
    assert n['evaluated']==1 and n['requested']==2 and n['accuracy']==1
    cn = next(r for r in result['interference'] if r['checked_factor']=='C' and r['added_factor']=='N')
    assert cn['eligible_pairs']==1 and cn['potential_pairs']==2
    assert cn['delta_accuracy']==-1 and cn['flip_rate']==1
    assert cn['gains']==0 and cn['mcnemar_p']==1 and cn['delta_ci_low']==cn['delta_ci_high']==-1
    empty = compute_metrics(ps,jobs,[],[])
    assert empty['strict']['accuracy'] is None and empty['strict']['coverage']==0


def test_mcnemar_and_holm():
    assert mcnemar_p(0, 0) is None
    assert mcnemar_p(10, 0) == pytest.approx(2 / 2**10)
    assert mcnemar_p(3, 3) == 1
    assert holm([0.01, None, 0.04, 0.03]) == [pytest.approx(0.03), None, pytest.approx(0.06), pytest.approx(0.06)]


def test_nonadditivity_and_bootstrap():
    ps, jobs, labels, pairs = [], [], [], []
    sets = {'': [], 'n': ['N'], 'p': ['P'], 'np': ['N', 'P']}
    for obj in ['cup', 'vase', 'lamp']:
        for key, extra in sets.items():
            pid = f'{obj}{key}'
            ps.append({'prompt_id': pid, 'object': obj, 'active': ['C'] + extra, 'complexity': 1 + len(extra)})
            for seed in range(4):
                jobs.append({'image_id': f'{pid}_s{seed}', 'prompt_id': pid, 'seed': seed})
                labels.append({'image_id': f'{pid}_s{seed}', 'prompt_id': pid, 'seed': seed, 'object': True,
                               'C': key != 'np', 'N': True, 'P': True})
        pairs += [{'base_id': f'{obj}', 'added_id': f'{obj}n', 'added_factor': 'N'},
                  {'base_id': f'{obj}p', 'added_id': f'{obj}np', 'added_factor': 'N'}]
    result = compute_metrics(ps, jobs, labels, pairs, draws=200)
    row = next(r for r in result['nonadditivity'] if r['checked_factor'] == 'C')
    assert row['added_factors'] == 'N|P' and row['additive_prediction'] == 0
    assert row['observed_delta'] == row['interaction'] == -1 and row['ci_low'] == row['ci_high'] == -1
    summary = next(r for r in result['nonadditivity_summary'] if r['checked_factor'] == 'C')
    assert summary['k'] == 2 and summary['sets'] == 1 and summary['mean_interaction'] == -1
    cn = next(r for r in result['interference'] if r['checked_factor'] == 'C' and r['added_factor'] == 'N')
    assert cn['delta_accuracy'] == -0.5 and cn['flips'] == 12 and cn['gains'] == 0
    assert cn['delta_ci_low'] <= -0.5 <= cn['delta_ci_high']
    assert cn['mcnemar_p_holm'] >= cn['mcnemar_p']
    assert result['bootstrap']['objects'] == 3 and result['bootstrap']['seed'] == 0


def test_config_and_data_immutability(tmp_path):
    cfg=config('smoke'); prepare(cfg,tmp_path)
    assert prepare(cfg,tmp_path)['job_count']==8
    changed=copy.deepcopy(cfg); changed['generation']['steps']=31
    with pytest.raises(ValueError,match='Config or source'):
        verify_run(changed,tmp_path)
    with (tmp_path/'prompts.csv').open('a') as f: f.write('bad')
    with pytest.raises(ValueError,match='Prepared data changed'):
        verify_run(cfg,tmp_path)


def write_fixture(root, p, seed, q=None):
    key=f"{p['prompt_id']}_s{seed}"
    path=root/'images'/f'{key}.png'; path.parent.mkdir(exist_ok=True)
    Image.new('RGB',(512,512),'white').save(path)
    sha=hashlib.sha256(path.read_bytes()).hexdigest()
    put_record(root,'generation_records',key,{'image_id':key,'prompt_id':p['prompt_id'],'seed':seed,
        'path':f'images/{key}.png','status':'ok','sha256':sha,'seconds':1})
    put_record(root,'qwen_records',key,{'image_id':key,'image_sha256':sha,'status':'ok',
        'parsed':q if q is not None else {'object':True,**{f:True for f in p['active']}}})
    put_record(root,'owl_records',key,{'image_id':key,'image_sha256':sha,'status':'ok','width':512,
        'boxes':[{'box':[0,0,10,10],'score':.9}]*p['count']})
    return key,path


def test_resume_corruption_and_stale_evaluation(tmp_path):
    cfg=config('smoke'); prepare(cfg,tmp_path)
    p=load_prompts(tmp_path)[0]
    key,path=write_fixture(tmp_path,p,0)
    assert len(pending_jobs(cfg,tmp_path))==7
    assert valid_image(path,[512,512])
    path.write_bytes(b'broken')
    assert len(pending_jobs(cfg,tmp_path))==8
    with pytest.raises(ValueError,match='missing or corrupt'):
        analyze(cfg,tmp_path)
    Image.new('RGB',(512,512),'black').save(path)
    g=read_json(tmp_path/'generation_records'/f'{key}.json')
    g['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    put_record(tmp_path,'generation_records',key,g)
    assert export_labels(cfg,tmp_path)[0]['object'] is None


def test_generation_adapter_resume(tmp_path,monkeypatch):
    import attribute_interference.generation as generation
    cfg=config('smoke'); prepare(cfg,tmp_path)
    calls=[]
    class FakePipe:
        scheduler=types.SimpleNamespace(config={'name':'test scheduler'})
        @classmethod
        def from_pretrained(cls,*a,**kw): return cls()
        def enable_model_cpu_offload(self): pass
        def enable_vae_slicing(self): pass
        def enable_vae_tiling(self): pass
        def __call__(self,**kw):
            calls.append(kw)
            return types.SimpleNamespace(images=[Image.new('RGB',(512,512),'white')],nsfw_content_detected=[False])
    class Generator:
        def __init__(self,device): pass
        def manual_seed(self,seed): return seed
    fake_torch=types.SimpleNamespace(float16='float16',Generator=Generator,
        cuda=types.SimpleNamespace(synchronize=lambda:None,empty_cache=lambda:None,OutOfMemoryError=MemoryError))
    monkeypatch.setattr(generation,'require_cuda',lambda:fake_torch)
    monkeypatch.setattr(generation,'record_environment',lambda *a:None)
    monkeypatch.setattr(generation,'model_revision',lambda *a:'fixture-sha')
    monkeypatch.setitem(sys.modules,'diffusers',types.SimpleNamespace(StableDiffusionPipeline=FakePipe,StableDiffusionXLPipeline=FakePipe))
    generate(cfg,tmp_path,limit=2)
    assert len(calls)==2 and len(pending_jobs(cfg,tmp_path))==6
    generate(cfg,tmp_path)
    assert len(calls)==8 and not pending_jobs(cfg,tmp_path)
    generate(cfg,tmp_path)
    assert len(calls)==8
    assert all(c['generator']==0 for c in calls)
    assert len((tmp_path/'generations.jsonl').read_text().splitlines())==8


def test_analysis_artifacts(tmp_path):
    cfg=config('smoke'); prepare(cfg,tmp_path)
    for p in load_prompts(tmp_path):
        write_fixture(tmp_path,p,0,{'object':False})
    result=analyze(cfg,tmp_path)
    assert result['run']['complete_generation']
    assert result['strict']['accuracy']==0
    assert len(read_csv(tmp_path/'analysis'/'failure_cases.csv'))==6
    assert (tmp_path/'analysis'/'interference.png').stat().st_size>0
    assert (tmp_path/'analysis'/'nonadditivity.csv').exists()


def test_review_and_freeze(tmp_path):
    cfg=config('pilot_sdxl'); prepare(cfg,tmp_path)
    # Synthetic fixtures only; never exported outside pytest's temporary directory.
    for p in load_prompts(tmp_path):
        for seed in [0,1]: write_fixture(tmp_path,p,seed)
    for role in ['qwen','owl']:
        atomic_json(tmp_path/'models'/f'{role}.json',{'checkpoint':cfg['evaluation'][role+'_checkpoint'],
            'requested_revision':'main','resolved_revision':'fixture-sha'})
        atomic_json(tmp_path/'environment'/f'{role}.json', {'python':'fixture','packages':{}})
    export_annotations(cfg,tmp_path)
    with pytest.raises(ValueError,match='preserving human edits'):
        export_annotations(cfg,tmp_path)
    with pytest.raises(ValueError,match='Missing reviewer'):
        freeze_evaluator(cfg,tmp_path)
    from attribute_interference.storage import write_csv
    path=tmp_path/'review'/'annotations.csv'
    rows=read_csv(path)
    complexities=set()
    ps={p['prompt_id']:p for p in load_prompts(tmp_path)}
    for r in rows:
        p=ps[r['prompt_id']]; complexities.add(p['complexity'])
        r['reviewer']='test fixture'
        for f in ['object']+p['active']: r['human_'+f]='true'
    assert complexities==set(range(6))
    write_csv(path,rows)
    freeze_evaluator(cfg,tmp_path)
    main=config(); target=tmp_path/'main'
    prepare(main,target)
    frozen=tmp_path/'evaluator_freeze.json'
    ensure_freeze(main,target,frozen)
    changed=copy.deepcopy(main); changed['evaluation']['threshold']=.4
    with pytest.raises(ValueError,match='differs'):
        ensure_freeze(changed,target,frozen)
    with pytest.raises(ValueError,match='requires --frozen'):
        ensure_freeze(main,target,None)


def test_evaluation_retry_cache_and_rgb(tmp_path, monkeypatch):
    import attribute_interference.evaluation as evaluation
    cfg=config('smoke'); prepare(cfg,tmp_path)
    prompts=load_prompts(tmp_path)
    for p in prompts: write_fixture(tmp_path,p,0)
    # Remove fixture labels to exercise actual evaluation orchestration.
    for folder in ['qwen_records','owl_records']:
        for path in (tmp_path/folder).glob('*.json'): path.unlink()
    calls=[]
    fail_once={prompts[0]['prompt_id']}
    class FakeQwen:
        def __init__(self,*args): pass
        def __call__(self,image,p):
            assert image.mode=='RGB'
            calls.append(('qwen',p['prompt_id']))
            if p['prompt_id'] in fail_once:
                fail_once.remove(p['prompt_id'])
                raise ValueError('simulated evaluator failure')
            return {'raw':{},'parsed':{'object':True,**{f:True for f in p['active']}}}
    class FakeOwl:
        def __init__(self,*args): pass
        def __call__(self,image,p):
            calls.append(('owl',p['prompt_id']))
            return {'boxes':[],'width':image.width,'height':image.height}
    torch=types.SimpleNamespace(cuda=types.SimpleNamespace(OutOfMemoryError=MemoryError,empty_cache=lambda:None))
    monkeypatch.setattr(evaluation,'require_cuda',lambda:torch)
    monkeypatch.setattr(evaluation,'record_environment',lambda *a:'fixture-environment')
    monkeypatch.setattr(evaluation,'QwenEvaluator',FakeQwen)
    monkeypatch.setattr(evaluation,'OwlEvaluator',FakeOwl)
    with pytest.raises(RuntimeError,match='1 evaluator jobs failed'):
        evaluation.evaluate(cfg,tmp_path)
    before=len(calls)
    evaluation.evaluate(cfg,tmp_path)
    assert len(calls)==before+1
    evaluation.evaluate(cfg,tmp_path)
    assert len(calls)==before+1
    assert sum(role=='owl' for role,_ in calls)==4
    labels=export_labels(cfg,tmp_path)
    assert len(labels)==8 and all(r['object'] is True for r in labels)


def test_main_requires_four_seeds(tmp_path):
    import yaml
    cfg=config(); cfg['dataset']['seeds']=[0]
    path=tmp_path/'bad.yaml'; path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError,match='four seeds'):
        load_config(path)


def test_hardware_change_recorded_software_change_rejected(tmp_path, monkeypatch):
    import attribute_interference.storage as storage
    monkeypatch.setattr(storage.importlib.metadata,'version',lambda name:'fixture-version')
    cuda=types.SimpleNamespace(get_device_name=lambda index:'GPU A')
    torch=types.SimpleNamespace(cuda=cuda,version=types.SimpleNamespace(cuda='12'))
    first=storage.record_environment(tmp_path,'qwen',torch)
    cuda.get_device_name=lambda index:'GPU B'
    second=storage.record_environment(tmp_path,'qwen',torch)
    assert first!=second
    assert len(list((tmp_path/'environment'/'sessions').glob('*.json')))==2
    monkeypatch.setattr(storage.importlib.metadata,'version',lambda name:'changed-version')
    with pytest.raises(ValueError,match='Software versions changed'):
        storage.record_environment(tmp_path,'qwen',torch)


@pytest.mark.parametrize('mode,offload', [('nf4',False),('none',True)])
def test_qwen_loading_quantization(tmp_path, monkeypatch, mode, offload):
    import attribute_interference.evaluation as evaluation
    captured={}
    class BnbConfig:
        def __init__(self,**kwargs): self.options=kwargs
    class Model:
        @classmethod
        def from_pretrained(cls,checkpoint,**kwargs): captured.update(kwargs); return cls()
        def eval(self): return self
    class Processor:
        @classmethod
        def from_pretrained(cls,*a,**kw): return cls()
    monkeypatch.setitem(sys.modules,'transformers',types.SimpleNamespace(
        AutoProcessor=Processor,Qwen2_5_VLForConditionalGeneration=Model,BitsAndBytesConfig=BnbConfig))
    monkeypatch.setattr(evaluation,'model_revision',lambda *a:'fixture-sha')
    cfg=config(); cfg['evaluation'].update(qwen_quantization=mode,qwen_cpu_offload=offload)
    torch=types.SimpleNamespace(float16='fp16',cuda=types.SimpleNamespace(
        get_device_properties=lambda i:types.SimpleNamespace(total_memory=16*1024**3)))
    evaluation.QwenEvaluator(cfg,tmp_path,torch)
    assert captured['low_cpu_mem_usage'] is True
    assert captured['torch_dtype']=='fp16'
    if mode=='nf4':
        assert captured['device_map']=={'':0}
        assert 'max_memory' not in captured and 'offload_folder' not in captured
        assert captured['quantization_config'].options==dict(load_in_4bit=True,
            bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype='fp16',bnb_4bit_use_double_quant=True)
    else:
        assert captured['device_map']=='auto' and 'quantization_config' not in captured


def test_quantization_changes_evaluator_signature_and_validates(tmp_path):
    import yaml
    from attribute_interference.evaluation import evaluator_signature
    cfg=config(); other=copy.deepcopy(cfg)
    other['evaluation'].update(qwen_quantization='none',qwen_cpu_offload=True)
    assert evaluator_signature(cfg)!=evaluator_signature(other)
    cfg['evaluation']['qwen_cpu_offload']=True
    path=tmp_path/'invalid.yaml'; path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError,match='NF4 Qwen'):
        load_config(path)


def test_reuse_images_without_labels(tmp_path):
    from attribute_interference.migration import reuse_generation
    new=config('smoke'); old=copy.deepcopy(new)
    old['evaluation'].update(qwen_quantization='none',qwen_cpu_offload=True)
    source,target=tmp_path/'old',tmp_path/'nf4'
    prepare(old,source); prepare(new,target)
    p=load_prompts(source)[0]; key,image=write_fixture(source,p,0)
    before=(source/'run.json').read_bytes()
    reuse_generation(new,target,source)
    assert (target/'images'/image.name).read_bytes()==image.read_bytes()
    assert (source/'run.json').read_bytes()==before
    assert not (target/'qwen_records').exists() and not (target/'owl_records').exists()
    assert not (target/'evaluator_freeze.json').exists()
    assert len(pending_jobs(new,target))==7
    verify_run(new,target)
    with pytest.raises(ValueError,match='already contains results'):
        reuse_generation(new,target,source)


def test_reuse_rejects_changed_generation(tmp_path):
    from attribute_interference.migration import reuse_generation
    cfg=config('smoke'); prepare(cfg,tmp_path/'old')
    cfg['generation']['steps']=20; prepare(cfg,tmp_path/'new')
    with pytest.raises(ValueError,match='generation settings differ'):
        reuse_generation(cfg,tmp_path/'new',tmp_path/'old')
