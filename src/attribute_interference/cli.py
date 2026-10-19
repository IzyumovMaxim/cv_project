"""Command-line entry point; inference dependencies load only when needed."""
import argparse
import sys
from .storage import load_config


def main(argv=None):
    parser = argparse.ArgumentParser(description='Text-to-image attribute interference pipeline')
    parser.add_argument('command', choices=['prepare', 'reuse-generation', 'generate', 'evaluate', 'export-annotations', 'freeze-evaluator', 'analyze'])
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--limit', type=int, help='Maximum pending inference jobs in this invocation; does not change the dataset')
    parser.add_argument('--backend', choices=['all', 'qwen', 'owl'], default='all')
    parser.add_argument('--frozen', help='Reviewed pilot evaluator_freeze.json; required for main evaluation')
    parser.add_argument('--count', type=int, default=24, help='Human review sample size')
    parser.add_argument('--from-run', help='Existing run to copy verified generations from; never copies evaluations')
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    try:
        cfg = load_config(args.config)
        if args.command == 'prepare':
            from .prompts import prepare
            prepare(cfg, args.output)
        elif args.command == 'reuse-generation':
            if not args.from_run:
                parser.error('reuse-generation requires --from-run')
            from .migration import reuse_generation
            reuse_generation(cfg, args.output, args.from_run)
        elif args.command == 'generate':
            from .generation import generate
            generate(cfg, args.output, args.limit)
        elif args.command == 'evaluate':
            from .evaluation import evaluate
            evaluate(cfg, args.output, args.backend, args.limit, args.frozen)
        elif args.command == 'export-annotations':
            from .annotation import export_annotations
            export_annotations(cfg, args.output, args.count)
        elif args.command == 'freeze-evaluator':
            from .annotation import freeze_evaluator
            freeze_evaluator(cfg, args.output)
        else:
            from .analysis import analyze
            analyze(cfg, args.output)
    except (ValueError, RuntimeError, FileNotFoundError, ImportError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
