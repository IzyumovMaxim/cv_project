"""Sequential, resumable diffusion inference; one RNG initialization per image."""
import hashlib
import time
from pathlib import Path
from .prompts import load_prompts
from .storage import (atomic_json, export_manifest, model_revision, put_record,
                      read_csv, record_environment, records, require_cuda, valid_image, verify_run)


def pending_jobs(cfg, root):
    previous = records(root, 'generation_records')
    size = [cfg['generation']['width'], cfg['generation']['height']]
    pending = []
    for job in read_csv(Path(root) / 'jobs.csv'):
        record = previous.get(job['image_id'], {})
        if record.get('status') == 'ok' and valid_image(
                Path(root) / record['path'], size, record.get('sha256')):
            continue
        pending.append(job)
    return pending


def generate(cfg, root, limit=None):
    verify_run(cfg, root)
    jobs = pending_jobs(cfg, root)
    if limit:
        jobs = jobs[:limit]
    if not jobs:
        export_manifest(root)
        print('All generation jobs are complete.')
        return
    torch = require_cuda()
    from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
    environment_id = record_environment(root, 'generation', torch)
    g = cfg['generation']
    revision = model_revision(root, 'generator', g['checkpoint'], g['revision'])
    cls = StableDiffusionXLPipeline if g['family'] == 'sdxl' else StableDiffusionPipeline
    pipe = cls.from_pretrained(g['checkpoint'], revision=revision,
                              torch_dtype=torch.float16, use_safetensors=True)
    if g['cpu_offload']:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to('cuda')
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()
    atomic_json(Path(root) / 'scheduler.json', dict(pipe.scheduler.config))
    prompts = {p['prompt_id']: p for p in load_prompts(root)}
    images = Path(root) / 'images'
    images.mkdir(exist_ok=True)
    failed = 0
    try:
        for i, job in enumerate(jobs, 1):
            key = job['image_id']
            row = {**job, 'seed': int(job['seed']), 'checkpoint': g['checkpoint'],
                   'resolved_revision': revision, 'environment_id': environment_id,
                   'settings': g, 'path': f'images/{key}.png'}
            start = time.perf_counter()
            try:
                torch.cuda.synchronize()
                start = time.perf_counter()
                result = pipe(prompt=prompts[job['prompt_id']]['prompt'],
                              generator=torch.Generator(device='cpu').manual_seed(int(job['seed'])),
                              width=g['width'], height=g['height'],
                              num_inference_steps=g['steps'], guidance_scale=g['guidance_scale'])
                torch.cuda.synchronize()
                row['seconds'] = time.perf_counter() - start
                flags = getattr(result, 'nsfw_content_detected', None)
                if flags and flags[0]:
                    raise RuntimeError('Safety checker flagged output; not scored as an attribute failure.')
                path = Path(root) / row['path']
                tmp = path.with_suffix('.tmp.png')
                result.images[0].save(tmp)
                tmp.replace(path)
                row.update(status='ok', sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            except Exception as exc:
                failed += 1
                row.update(status='error', error=f'{type(exc).__name__}: {exc}',
                           seconds=time.perf_counter() - start)
                if isinstance(exc, torch.cuda.OutOfMemoryError):
                    put_record(root, 'generation_records', key, row)
                    raise RuntimeError('GPU memory exhausted. Progress saved; retry with the configured offload or a larger GPU.') from exc
            put_record(root, 'generation_records', key, row)
            print(f'[{i}/{len(jobs)}] {key}: {row["status"]}', flush=True)
    finally:
        export_manifest(root)
        del pipe
        torch.cuda.empty_cache()
    if failed:
        raise RuntimeError(f'{failed} generation jobs failed. Inspect generation_records and rerun to retry.')
