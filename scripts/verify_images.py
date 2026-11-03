"""Compare generated images with the published SHA-256 hashes."""
import argparse
import csv
import hashlib
import sys
from collections import Counter
from pathlib import Path


def read_hashes(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(run, rows):
    outcome = Counter()
    different = []
    for row in rows:
        path = Path(run) / 'images' / f"{row['image_id']}.png"
        if not row['sha256']:
            outcome['filtered_in_original_run'] += 1
        elif not path.exists():
            outcome['missing'] += 1
        elif file_hash(path) == row['sha256']:
            outcome[f"matched_{row['gpu']}"] += 1
        else:
            outcome[f"different_{row['gpu']}"] += 1
            different.append(row['image_id'])
    return outcome, different


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, help='Run directory with an images folder')
    parser.add_argument('--hashes', required=True, help='CSV file from results/image_hashes')
    args = parser.parse_args(argv)

    outcome, different = compare(args.run, read_hashes(args.hashes))
    for name, count in sorted(outcome.items()):
        print(f'{name}: {count}')
    for image_id in different[:10]:
        print(f'different image: {image_id}')
    return 1 if different else 0


if __name__ == '__main__':
    sys.exit(main())
