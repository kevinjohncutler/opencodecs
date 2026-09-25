"""Validate and report the source-grounded cross-codec optimization catalog.

This verifies coverage, references, dependencies, and evidence locations. It
cannot prove a performance benefit or a native feature's semantics; those need
implementation tests and measurements before runtime capability flags change.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent.parent


def load(path):
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib
    with Path(path).open('rb') as stream:
        return tomllib.load(stream)


def validate(catalog, codec_names, root):
    """Return diagnostics without importing optional codec libraries."""
    root = Path(root)
    errors = []
    if catalog.get('schema') != 1:
        errors.append('unsupported pipeline catalog schema')
    groups = {key: catalog.get(key, []) for key in
              ('family', 'adapter', 'work', 'finding')}
    for kind, records in groups.items():
        ids = [r.get('id') for r in records]
        if any(not isinstance(i, str) or not i for i in ids):
            errors.append(f'{kind}: every entry needs a nonempty id')
        for value, count in Counter(ids).items():
            if count > 1:
                errors.append(f'duplicate {kind} id: {value}')
    members = [m for r in groups['family'] for m in r.get('members', [])]
    for value, count in Counter(members).items():
        if count > 1:
            errors.append(f'codec belongs to multiple families: {value}')
    missing, unknown = set(codec_names) - set(members), set(members) - set(codec_names)
    if missing:
        errors.append(f'unclassified codecs: {sorted(missing)}')
    if unknown:
        errors.append(f'unknown family codecs: {sorted(unknown)}')
    targets = set(codec_names) | {f"adapter:{r['id']}" for r in groups['adapter']}
    work_by_id = {r['id']: r for r in groups['work']}
    addressed = set()
    texts = {}
    for kind, records in groups.items():
        for record in records:
            label = f"{kind}:{record.get('id')}"
            if not record.get('evidence'):
                errors.append(f'{label}: missing source evidence')
            evidence = record.get('evidence', []) + record.get('implementation_evidence', [])
            for item in evidence:
                path, _, symbol = item.partition(':')
                relative = Path(path)
                # anchor, not is_absolute(): on Windows '/etc/passwd' has a root
                # but no drive, so is_absolute() calls it relative.
                if relative.anchor or '..' in relative.parts:
                    errors.append(f'{label}: evidence must be repository-relative: {item}')
                    continue
                if not (root / path).is_file():
                    errors.append(f'{label}: missing evidence file: {path}')
                    continue
                if symbol:
                    if path not in texts:
                        texts[path] = (root / path).read_text(encoding="utf-8")
                    # Check named tokens, not unstable line numbers. This
                    # checks location drift, not the claimed behavior.
                    for token in symbol.split('.'):
                        if not re.search(r'\b' + re.escape(token) + r'\b', texts[path]):
                            errors.append(f'{label}: missing evidence symbol: {item}')
                            break
            if kind == 'work':
                if record.get('state') not in ('ready', 'design', 'investigate', 'complete'):
                    errors.append(f'{label}: invalid state')
                if record.get('state') == 'complete' and not record.get('implementation_evidence'):
                    errors.append(f'{label}: completed work needs implementation evidence')
                if record.get('priority') not in (0, 1, 2, 3):
                    errors.append(f'{label}: invalid priority')
                for field in ('title', 'current', 'action', 'limits', 'acceptance'):
                    if not record.get(field):
                        errors.append(f'{label}: missing {field}')
                adopted = [t for field in ('pilots', 'followups', 'investigate')
                           for t in record.get(field, [])]
                if len(adopted) != len(set(adopted)):
                    errors.append(f'{label}: duplicate target assessment')
                for target in adopted:
                    if target not in targets:
                        errors.append(f'{label}: unknown target {target}')
                addressed.update(adopted)
                for dependency in record.get('depends_on', []):
                    if dependency not in work_by_id:
                        errors.append(f'{label}: unknown dependency {dependency}')
            if kind == 'finding':
                if record.get('confidence') not in ('reproduced', 'source_confirmed', 'hypothesis'):
                    errors.append(f'{label}: invalid confidence')
                for field in ('summary', 'reproduction', 'targets', 'blocks'):
                    if not record.get(field):
                        errors.append(f'{label}: missing {field}')
                for target in record.get('targets', []):
                    if target not in targets:
                        errors.append(f'{label}: unknown target {target}')
                for work in record.get('blocks', []):
                    if work not in work_by_id:
                        errors.append(f'{label}: unknown blocked work {work}')
    # A family assignment alone must not hide an unassessed codec. Work items
    # can explicitly say investigate, which is not a promise of a benefit.
    if targets - addressed:
        errors.append(f'targets missing work assessment: {sorted(targets - addressed)}')
    visiting, visited = set(), set()

    def walk(name):
        if name in visiting:
            errors.append(f'cyclic work dependency at {name}')
            return
        if name in visited or name not in work_by_id:
            return
        visiting.add(name)
        for child in work_by_id[name].get('depends_on', []):
            walk(child)
        visiting.remove(name)
        visited.add(name)

    for name in work_by_id:
        walk(name)
    return errors


def verify(root=ROOT):
    root = Path(root)
    try:
        catalog = load(root / 'pipeline_catalog.toml')
        names = [c['name'] for c in load(root / 'capabilities.toml')['codec']]
        return validate(catalog, names, root)
    except (OSError, ValueError, KeyError) as exc:
        return [f'could not load pipeline catalog: {exc}']


def report(catalog):
    """Print a reusable Markdown worklist; the catalog remains authoritative."""
    count = sum(len(f['members']) for f in catalog['family'])
    print(f"{count} registered codecs, {len(catalog['adapter'])} direct adapters, "
          f"{len(catalog['work'])} shared work packages.\n")
    print('| Priority | Work | State | First targets |')
    print('|---|---|---|---|')
    for row in sorted(catalog['work'], key=lambda r: r['priority']):
        first = ', '.join(row['pilots'] or row['investigate'])
        print(f"| P{row['priority']} | {row['title']} | {row['state']} | {first} |")
    print('\nReady means source-grounded implementation work, not shipped support '
          'or a measured speedup. Complete closes the stated implementation or '
          'assessment. Investigated targets can still need new native features; '
          'each record documents those boundaries.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('verify', 'report'))
    args = parser.parse_args()
    errors = verify()
    if errors:
        for error in errors:
            print(f'  BAD pipeline: {error}')
        return 1
    if args.command == 'report':
        report(load(ROOT / 'pipeline_catalog.toml'))
    else:
        print('Pipeline catalog: complete coverage, valid references, no dependency cycles')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
