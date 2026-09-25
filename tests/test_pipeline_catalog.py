"""Guard audit coverage and dependencies without requiring native codecs."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    'pipeline_catalog_checker', ROOT / 'ci' / 'check_pipeline_catalog.py')
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


@pytest.fixture
def catalog():
    return checker.load(ROOT / 'pipeline_catalog.toml')


@pytest.fixture
def names():
    return [c['name'] for c in checker.load(ROOT / 'capabilities.toml')['codec']]


def test_current_catalog_is_complete(catalog, names):
    assert checker.validate(catalog, names, ROOT) == []


def test_new_codec_requires_an_assessment(catalog, names):
    errors = checker.validate(catalog, names + ['new_codec'], ROOT)
    assert any('unclassified codecs' in e and 'new_codec' in e for e in errors)


def test_family_membership_is_unique(catalog, names):
    catalog['family'][1]['members'].append(catalog['family'][0]['members'][0])
    assert any('multiple families' in e for e in checker.validate(catalog, names, ROOT))


def test_family_alone_does_not_count_as_work_assessment(catalog, names):
    victim = names[0]
    for row in catalog['work']:
        for field in ('pilots', 'followups', 'investigate'):
            row[field] = [t for t in row[field] if t != victim]
    assert any('missing work assessment' in e and victim in e
               for e in checker.validate(catalog, names, ROOT))


@pytest.mark.parametrize('field,value,expected', [
    ('pilots', ['adapter:invented'], 'unknown target'),
    ('depends_on', ['nonexistent'], 'unknown dependency'),
    ('state', 'shipped-maybe', 'invalid state'),
    ('acceptance', [], 'missing acceptance'),
    ('evidence', ['src/does_not_exist.py:decode'], 'missing evidence file'),
    ('evidence', ['src/opencodecs/core/parallel.py:invented_symbol'], 'missing evidence symbol'),
    ('evidence', ['/etc/passwd'], 'repository-relative'),
])
def test_bad_work_records_are_rejected(catalog, names, field, value, expected):
    catalog['work'][0][field] = value
    assert any(expected in e for e in checker.validate(catalog, names, ROOT))


def test_dependency_cycles_are_rejected(catalog, names):
    a, b = catalog['work'][:2]
    a['depends_on'] = [b['id']]
    b['depends_on'] = [a['id']]
    assert any('cyclic work dependency' in e for e in checker.validate(catalog, names, ROOT))


def test_completed_work_requires_reviewable_evidence(catalog, names):
    row = catalog['work'][0]
    row['state'] = 'complete'
    row.pop('implementation_evidence', None)
    assert any('completed work needs implementation evidence' in error
               for error in checker.validate(catalog, names, ROOT))


def test_findings_cannot_block_unknown_work(catalog, names):
    catalog['finding'][0]['blocks'] = ['invented']
    assert any('unknown blocked work' in e for e in checker.validate(catalog, names, ROOT))


def test_report_distinguishes_candidates_from_shipped(catalog, capsys):
    checker.report(catalog)
    result = capsys.readouterr().out
    assert '60 registered codecs' in result
    assert '|---|---|---|---|' in result
    assert 'not shipped support' in result
