"""Evaluate generated images with OWLv2 and 8-bit Qwen, as in the final runs, and write labels and metrics."""
import argparse
import shutil
import sys
from pathlib import Path

import attribute_interference.analysis as analysis
import attribute_interference.evaluation as evaluation
import attribute_interference.storage as storage

QWEN_RECORDS = 'qwen_int8_records'
SKIP_MODULES = ['visual', 'lm_head']


def int8_records(kind):
    return QWEN_RECORDS if kind == 'qwen_records' else kind


def use_int8_records():
    read, write = storage.records, storage.put_record
    evaluation.records = analysis.records = lambda root, kind: read(root, int8_records(kind))
    evaluation.put_record = lambda root, kind, key, value: write(root, int8_records(kind), key, value)


def int8_init(self, cfg, root, torch):
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
    self.torch = torch
    settings = cfg['evaluation']
    revision = evaluation.model_revision(root, 'qwen', settings['qwen_checkpoint'], settings['qwen_revision'])
    self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        settings['qwen_checkpoint'], revision=revision, torch_dtype=torch.float16, attn_implementation='sdpa',
        device_map={'': 0},
        quantization_config=BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=SKIP_MODULES)).eval()
    self.processor = AutoProcessor.from_pretrained(settings['qwen_checkpoint'], revision=revision,
                                                   min_pixels=256 * 28 * 28, max_pixels=settings['max_pixels'])
    self.max_tokens = settings['max_new_tokens']


def write_metrics(cfg, root):
    analysis.analyze(cfg, root)
    target = root / 'analysis_qwen_int8'
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(root / 'analysis', target)
    shutil.copy(root / 'labels.csv', target / 'labels.csv')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--run', required=True, help='Run directory with generated images')
    parser.add_argument('--limit', type=int, help='Evaluate only this many pending images and skip metrics')
    args = parser.parse_args(argv)

    cfg = storage.load_config(args.config)
    root = Path(args.run)
    evaluation.ensure_freeze = lambda *_: None
    evaluation.QwenEvaluator.__init__ = int8_init
    use_int8_records()
    try:
        evaluation.evaluate(cfg, root, 'owl', args.limit)
        evaluation.evaluate(cfg, root, 'qwen', args.limit)
        if not args.limit:
            write_metrics(cfg, root)
    except (ValueError, RuntimeError, FileNotFoundError, ImportError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1
    print(f'Labels: {root / "labels.csv"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
