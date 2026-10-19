"""Raw evaluator outputs, conservative parsing, detector NMS and label fusion."""
import gc
import re
from pathlib import Path
from .prompts import FACTORS, load_prompts, plural
from .storage import (atomic_json, digest, model_revision, put_record, read_json,
                      record_environment, records, require_cuda, source_hash, valid_image, verify_run, write_csv)

QUESTION_VERSION = '1'


def parse_yes_no(text):
    value = text.strip().lower().strip('.! ')
    return {'yes': True, 'no': False}.get(value)


def parse_count(text):
    value = text.strip().lower().strip('.! ')
    words = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']
    if value in words:
        return words.index(value)
    return int(value) if re.fullmatch(r'\d{1,3}', value) else None


def iou(a, b):
    inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area_a, area_b = (a[2]-a[0])*(a[3]-a[1]), (b[2]-b[0])*(b[3]-b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0


def filter_boxes(boxes, scores, threshold, nms_iou, width, height):
    candidates = []
    for box, score in zip(boxes, scores):
        if score < threshold:
            continue
        x1, y1, x2, y2 = box
        clipped = [max(0, min(width, x1)), max(0, min(height, y1)),
                   max(0, min(width, x2)), max(0, min(height, y2))]
        if clipped[2] > clipped[0] and clipped[3] > clipped[1]:
            candidates.append((clipped, score))
    kept = []
    for box, score in sorted(candidates, key=lambda x: -x[1]):
        if all(iou(box, old['box']) <= nms_iou for old in kept):
            kept.append({'box': box, 'score': score})
    return kept


def correct_position(boxes, width, side):
    if not boxes:
        return False
    centers = [(b['box'][0] + b['box'][2]) / 2 for b in boxes]
    return all(x < width/2 for x in centers) if side == 'left' else all(x > width/2 for x in centers)


def questions(p):
    noun = plural(p['object'])
    yesno = ' Answer only Yes or No.'
    q = {'object': f"Is there at least one {p['object']} in the image?" + yesno}
    templates = {
        'C': f"Are all visible {noun} {p['color']}?",
        'M': f"Are all visible {noun} made of {p['material']} material?",
        'P': f"Are the centers of all visible {noun} in the {p['position']} half of the image?",
        'O2': f"Is there a {p['second_color']} {p['second_object']} in the image?",
    }
    for f in p['active']:
        q[f] = (f"How many {noun} are visible in the image? Answer only with one nonnegative integer."
                if f == 'N' else templates[f] + yesno)
    return q


def fused_labels(p, qwen, owl):
    q = qwen.get('parsed', {}) if qwen and qwen.get('status') == 'ok' else {}
    present = q.get('object')
    result = {'object': present}
    for factor in p['active']:
        if present is False:
            result[factor] = False
        elif present is None:
            result[factor] = None
        elif factor in {'N', 'P'}:
            if not owl or owl.get('status') != 'ok':
                result[factor] = None
            elif factor == 'N':
                result[factor] = len(owl['boxes']) == p['count']
            else:
                result[factor] = correct_position(owl['boxes'], owl['width'], p['position'])
        else:
            result[factor] = q.get(factor)
    return result


def evaluator_signature(cfg):
    return digest({'evaluation': cfg['evaluation'], 'question_version': QUESTION_VERSION})


def ensure_freeze(cfg, root, frozen):
    if cfg['dataset']['split'] != 'main':
        return
    if not frozen:
        raise ValueError('Main evaluation requires --frozen pointing to a reviewed pilot evaluator_freeze.json.')
    record = read_json(frozen)
    if record['source_hash'] != source_hash():
        raise ValueError('Evaluator source differs from the reviewed pilot. Repeat development validation.')
    if record['evaluator_signature'] != evaluator_signature(cfg):
        raise ValueError('Evaluator differs from the reviewed pilot. Freeze the matching settings first.')
    saved = Path(root) / 'evaluator_freeze.json'
    if saved.exists() and read_json(saved) != record:
        raise ValueError('A different evaluator freeze was already used in this run.')
    atomic_json(saved, record)
    # Copy pinned evaluator revisions from the development run.
    for role, model in record['models'].items():
        path = Path(root) / 'models' / f'{role}.json'
        if path.exists() and read_json(path) != model:
            raise ValueError('Evaluator model revision differs from the frozen development model.')
        atomic_json(path, model)
    for role, environment in record['environments'].items():
        path = Path(root) / 'environment' / f'{role}.json'
        if path.exists():
            current = read_json(path)
            if any(current[k] != environment[k] for k in ['python', 'packages']):
                raise ValueError('Evaluator software differs from the frozen development environment.')
        else:
            atomic_json(path, environment)


class QwenEvaluator:
    def __init__(self, cfg, root, torch):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig
        self.torch = torch
        e = cfg['evaluation']
        revision = model_revision(root, 'qwen', e['qwen_checkpoint'], e['qwen_revision'])
        kwargs = {'device_map': 'auto', 'torch_dtype': torch.float16,
                  'attn_implementation': 'sdpa', 'revision': revision,
                  'low_cpu_mem_usage': True}
        mode = e.get('qwen_quantization', 'none')
        if mode == 'nf4':
            kwargs['device_map'] = {'': 0}
            kwargs['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
        elif e['qwen_cpu_offload']:
            available = int(torch.cuda.get_device_properties(0).total_memory / 1024**3)
            kwargs['max_memory'] = {0: f'{max(1, available - 3)}GiB', 'cpu': e['cpu_memory']}
            kwargs['offload_folder'] = str(Path(root) / '.offload')
        else:
            kwargs['device_map'] = {'': 0}
        print(f'Loading Qwen: quantization={mode}, device_map={kwargs["device_map"]}', flush=True)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(e['qwen_checkpoint'], **kwargs).eval()
        print('Qwen loaded; starting image evaluation.', flush=True)
        self.processor = AutoProcessor.from_pretrained(e['qwen_checkpoint'], revision=revision,
                                                       min_pixels=256*28*28, max_pixels=e['max_pixels'])
        self.max_tokens = e['max_new_tokens']

    def __call__(self, image, p):
        raw, parsed = {}, {}
        for factor, question in questions(p).items():
            messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': question}]}]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text], images=[image], padding=True, return_tensors='pt').to('cuda')
            with self.torch.inference_mode():
                out = self.model.generate(**inputs, max_new_tokens=self.max_tokens, do_sample=False)
            answer = self.processor.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
            raw[factor] = {'question': question, 'answer': answer}
            value = parse_count(answer) if factor == 'N' else parse_yes_no(answer)
            parsed[factor] = (value == p['count'] if value is not None else None) if factor == 'N' else value
            if factor == 'N':
                raw[factor]['parsed_count'] = value
        return {'raw': raw, 'parsed': parsed}


class OwlEvaluator:
    def __init__(self, cfg, root, torch):
        from transformers import Owlv2ForObjectDetection, Owlv2Processor
        self.torch, self.e = torch, cfg['evaluation']
        revision = model_revision(root, 'owl', self.e['owl_checkpoint'], self.e['owl_revision'])
        self.processor = Owlv2Processor.from_pretrained(self.e['owl_checkpoint'], revision=revision)
        self.model = Owlv2ForObjectDetection.from_pretrained(self.e['owl_checkpoint'], revision=revision).to('cuda').eval()

    def __call__(self, image, p):
        inputs = self.processor(text=[[f"a photo of a {p['object']}"]], images=image, return_tensors='pt').to('cuda')
        with self.torch.inference_mode():
            outputs = self.model(**inputs)
        result = self.processor.post_process_grounded_object_detection(
            outputs, threshold=0.0, target_sizes=self.torch.tensor([[image.height, image.width]]))[0]
        boxes, scores = result['boxes'].detach().cpu().tolist(), result['scores'].detach().cpu().tolist()
        kept = filter_boxes(boxes, scores, self.e['threshold'], self.e['nms_iou'], image.width, image.height)
        return {'raw_boxes': boxes, 'raw_scores': scores, 'boxes': kept,
                'width': image.width, 'height': image.height}


def export_labels(cfg, root):
    qwen, owl = records(root, 'qwen_records'), records(root, 'owl_records')
    prompts = {p['prompt_id']: p for p in load_prompts(root)}
    rows = []
    for key, gen in records(root, 'generation_records').items():
        if gen['status'] != 'ok':
            continue
        p = prompts[gen['prompt_id']]
        q, o = qwen.get(key), owl.get(key)
        q = q if q and q.get('image_sha256') == gen['sha256'] else None
        o = o if o and o.get('image_sha256') == gen['sha256'] else None
        fused = fused_labels(p, q, o)
        row = {'image_id': key, 'prompt_id': p['prompt_id'], 'seed': gen['seed'],
               'path': gen['path'], 'complexity': p['complexity'],
               'qwen_status': q['status'] if q else 'pending', 'owl_status': o['status'] if o else 'pending'}
        for f in ['object'] + FACTORS:
            row[f] = fused.get(f)
            row[f'qwen_{f}'] = q.get('parsed', {}).get(f) if q else None
        row['owl_count'] = len(o['boxes']) if o and o['status'] == 'ok' else None
        row['owl_position'] = correct_position(o['boxes'], o['width'], p['position']) if o and o['status'] == 'ok' and 'P' in p['active'] else None
        rows.append(row)
    fields = ['image_id', 'prompt_id', 'seed', 'path', 'complexity', 'qwen_status', 'owl_status']
    for f in ['object'] + FACTORS:
        fields += [f, f'qwen_{f}']
    write_csv(Path(root) / 'labels.csv', rows, fields + ['owl_count', 'owl_position'])
    return rows


def evaluate(cfg, root, backend='all', limit=None, frozen=None):
    verify_run(cfg, root)
    ensure_freeze(cfg, root, frozen)
    from PIL import Image
    prompts = {p['prompt_id']: p for p in load_prompts(root)}
    generations = records(root, 'generation_records')
    failed = 0
    for role in (['qwen', 'owl'] if backend == 'all' else [backend]):
        previous = records(root, role + '_records')
        pending = []
        for key, gen in generations.items():
            if gen['status'] != 'ok':
                continue
            if role == 'owl' and not set(prompts[gen['prompt_id']]['active']) & {'N', 'P'}:
                continue
            old = previous.get(key, {})
            if old.get('status') == 'ok' and old.get('image_sha256') == gen['sha256']:
                continue
            pending.append((key, gen))
        if limit:
            pending = pending[:limit]
        if not pending:
            continue
        torch = require_cuda()
        environment_id = record_environment(root, role, torch)
        evaluator = (QwenEvaluator if role == 'qwen' else OwlEvaluator)(cfg, root, torch)
        try:
            for i, (key, gen) in enumerate(pending, 1):
                row = {'image_id': key, 'image_sha256': gen['sha256'], 'environment_id': environment_id}
                try:
                    path = Path(root) / gen['path']
                    if not valid_image(path, expected_hash=gen['sha256']):
                        raise ValueError('Image missing/corrupt/changed. Rerun generate before evaluating.')
                    with Image.open(path) as image:
                        row.update(evaluator(image.convert('RGB'), prompts[gen['prompt_id']]))
                    row['status'] = 'ok'
                except Exception as exc:
                    failed += 1
                    row.update(status='error', error=f'{type(exc).__name__}: {exc}')
                    if isinstance(exc, torch.cuda.OutOfMemoryError):
                        put_record(root, role + '_records', key, row)
                        raise RuntimeError('Evaluator GPU memory exhausted. Progress saved; use a larger GPU with unchanged settings.') from exc
                put_record(root, role + '_records', key, row)
                print(f'{role} [{i}/{len(pending)}] {key}: {row["status"]}', flush=True)
        finally:
            del evaluator
            gc.collect()
            torch.cuda.empty_cache()
            export_labels(cfg, root)
    export_labels(cfg, root)
    if failed:
        raise RuntimeError(f'{failed} evaluator jobs failed. Inspect raw records and rerun to retry.')
