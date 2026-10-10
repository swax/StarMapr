#!/usr/bin/env python3
"""Show cooldown savings and domains supplying accepted training images."""

import argparse
import csv
import json
from pathlib import Path

from training_history import TrainingHistory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json', action='store_true', help='Print the complete machine-readable report')
    parser.add_argument('--domains-csv', type=Path, help='Export all accepted source-page domains as CSV')
    args = parser.parse_args()
    report = TrainingHistory().summary()
    if args.domains_csv:
        args.domains_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.domains_csv.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=['domain', 'images', 'actors', 'successful_runs'])
            writer.writeheader()
            writer.writerows(report['page_domain'])
    if args.json:
        print(json.dumps(report, indent=2))
        return
    print(f"Failed runs starting cooldowns: {report['failed_runs']}")
    print(f"Training attempts skipped during cooldown: {report['cooldown_skips']}")
    print(f"Actors currently cooling down: {report['active_cooldowns']}")
    print(f"Successful training runs recorded: {report['successful_runs']}")
    print('\nSource-page domains (unique accepted images / actors / successful runs):')
    for row in report['page_domain']:
        print(f"  {row['domain']}: {row['images']} / {row['actors']} / {row['successful_runs']}")
    if not report['page_domain']:
        print('  None recorded yet.')
    print(f"Accepted image uses without a known source page: {report['accepted_image_uses_without_page_domain']}")
    for actor in report['actors']:
        if actor['cooling_down'] or actor['cooldown_skips']:
            print(f"  {actor['actor_name']}: {actor['cooldown_skips']} cooldown skips; "
                  f"retry after {actor['retry_after'] if actor['cooling_down'] else 'now'}")


if __name__ == '__main__':
    main()
