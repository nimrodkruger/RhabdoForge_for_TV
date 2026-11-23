"""Hardware tags: registration round-trip and calibration selection rules (no real data needed)."""
import json
from pathlib import Path

import numpy as np
import pytest

from rhabdoforge.hardware import assets as hw
from rhabdoforge.hardware.assets import AssetError, load_asset
from rhabdoforge.hardware.selection import (
    SelectionError, recording_hardware_tag, select_calibration, tag_status,
)

MISSING = object()


def _tv_tree(root: Path, head='left', rays_extra=None, rec_name='rec_src'):
    """A minimal TrilobiteVision layout with one calibration export for `head`."""
    b = root / f'analysis/run/calibration/{head}'
    b.mkdir(parents=True, exist_ok=True)
    rays = {'model': {'model': 'focused_plenoptic_virtual_camera_array'}, 'trial': 't', 'camera': head,
            'status': 'test', 'frame': {'units': 'mm'}, 'lenslets': [{'index': 0}]}
    rays.update(rays_extra or {})
    (b / 'rays.json').write_text(json.dumps(rays))
    np.save(b / 'rays.npy', np.full((2, 2, 6), len(json.dumps(rays)), np.float32))  # unique per export
    np.save(b / 'lenslet.npy', np.zeros((2, 2), np.int16))
    run = root / 'analysis/run'
    (run / 'run.json').write_text('{}')
    ref = run / f'references/rec-1/{head}'
    ref.mkdir(parents=True, exist_ok=True)
    for name in hw.REFERENCE_FILES:
        (ref / name).write_bytes(b'x' * 8)
    lat = run / f'lattice/rec-1/{head}'
    lat.mkdir(parents=True, exist_ok=True)
    (lat / 'roi.json').write_text('{}')
    rec = root / f'data/{rec_name}'
    rec.mkdir(parents=True, exist_ok=True)
    (rec / 'recording.json').write_text('{}')
    return root


def _register(tmp, version, head='left', rays_extra=None, tag=None, assigned_by='op'):
    src = _tv_tree(tmp / f'tv-{version}-{head}', head, rays_extra)
    m = hw.build_manifest(
        asset=f'cam-{head}', version=version, label='l', status='s', source_root=src,
        bundle=f'analysis/run/calibration/{head}', calibration_run='analysis/run', session_id='rec-1',
        head=head, source_recording='data/rec_src', provenance={}, hardware_tag=tag,
        tag_assignment={'assigned_by': assigned_by, 'note': 'bench log'} if tag else None)
    hw.write_manifest(m, tmp / 'manifests')
    asset = load_asset(f'cam-{head}@{version}', manifest_dir=tmp / 'manifests', store=tmp / 'store')
    assert asset.import_from(src)['ok']
    return m


def _recording(tmp, tags, name='rec'):
    """A recording journal with per-head tags; MISSING omits the field (legacy)."""
    d = tmp / name
    d.mkdir(parents=True, exist_ok=True)
    heads = {}
    for head, tag in tags.items():
        heads[head] = {'mla': {}} if tag is MISSING else {'mla': {}, 'hardware_tag': tag}
    (d / 'recording.json').write_text(json.dumps({'geometry': {'heads': heads},
                                                  'geometry_at_start': {'heads': heads}}))
    return d


def _select(tmp, rec, head='left', **kw):
    return select_calibration(rec, head, manifest_dir=tmp / 'manifests', store=tmp / 'store', **kw)


# Registration round-trip

def test_registration_takes_the_export_tag_and_its_assignment(tmp_path):
    m = _register(tmp_path, 'v1', rays_extra={
        'hardware_tag': 'rig1-L-r1',
        'hardware_tag_assignment': {'basis': 'unanimous contributing recordings', 'sessions': ['a', 'b']}})
    assert m['hardware_tag'] == 'rig1-L-r1'
    assert m['hardware_tag_assignment']['source'] == 'calibration export'
    assert m['hardware_tag_assignment']['sessions'] == ['a', 'b'] and m['registration_warnings'] == []
    assert load_asset('cam-left@v1', manifest_dir=tmp_path / 'manifests').hardware_tag == 'rig1-L-r1'


def test_explicit_registration_tag_wins_and_a_conflict_with_the_export_is_visible(tmp_path):
    m = _register(tmp_path, 'v1', rays_extra={'hardware_tag': 'export-tag'}, tag='operator-tag', assigned_by='N.K.')
    assert m['hardware_tag'] == 'operator-tag'
    a = m['hardware_tag_assignment']
    assert a['source'] == 'registration' and a['assigned_by'] == 'N.K.' and a['export_tag'] == 'export-tag'
    assert any('differs' in w for w in m['registration_warnings'])
    with pytest.raises(AssetError, match='who assigned'):
        _register(tmp_path, 'v2', tag='x', assigned_by='')


@pytest.mark.parametrize('extra,note', [({}, 'no hardware tag was recorded'),
                                        ({'hardware_tag': None}, 'export states no hardware tag')])
def test_legacy_or_unset_exports_register_untagged(tmp_path, extra, note):
    m = _register(tmp_path, 'v1', rays_extra=extra)
    assert m['hardware_tag'] is None and note in m['hardware_tag_assignment']['note']


def test_mixed_or_unknown_input_provenance_is_carried_into_the_manifest(tmp_path):
    m = _register(tmp_path, 'v1', rays_extra={
        'hardware_tag': None,
        'provenance': {'recordings': {'tag_warnings': [
            "contributing recordings disagree on the hardware tag: ['A', 'B']",
            "contributing recordings without a stated tag: ['rec-3']"]}}})
    assert m['hardware_tag'] is None
    assert len(m['registration_warnings']) == 2
    assert all(w.startswith('calibration export:') for w in m['registration_warnings'])


def test_re_registering_a_version_with_other_contents_or_tag_is_refused(tmp_path):
    m = _register(tmp_path, 'v1')
    hw.write_manifest(m, tmp_path / 'manifests')                       # identical: fine
    changed = {**m, 'hardware_tag': 'late-tag'}
    with pytest.raises(AssetError, match='already registered'):
        hw.write_manifest(changed, tmp_path / 'manifests')


def test_the_registered_left_room_manifest_is_preserved_untagged():
    a = load_asset('trilobite-left@left-room-20261105-scaled004')
    assert a.hardware_tag is None and a.manifest['head'] == 'left'
    assert a.calibration_digest == 'ab524d562420a904f6cd2ce073f399a24d1bf3da3ae383b5fe84e805b3e14270'
    assert a.content_digest == 'c4d1e1a39a7c343c78d24a3390a2a7f5f2cd2d0c3c715ac4d304fb597c77751a'


# Recording journals

def test_recording_tag_statuses_and_unknown_heads(tmp_path):
    rec = _recording(tmp_path, {'left': ' rig-L ', 'right': None, 'other': MISSING})
    assert recording_hardware_tag(rec, 'left') == {'tag_status': 'tagged', 'hardware_tag': 'rig-L',
                                                   'section': 'geometry_at_start'}
    assert recording_hardware_tag(rec, 'right')['tag_status'] == 'unset'
    assert recording_hardware_tag(rec, 'other')['tag_status'] == 'absent'
    with pytest.raises(SelectionError, match="no head 'middle'"):
        recording_hardware_tag(rec, 'middle')
    with pytest.raises(SelectionError, match='cannot read'):
        recording_hardware_tag(tmp_path / 'nowhere', 'left')


def test_missing_values_never_match():
    assert tag_status('A', 'A') == 'match'
    assert tag_status('A', 'B') == 'mismatch'
    assert tag_status(None, 'A') == 'recording_missing'
    assert tag_status('A', None) == 'calibration_missing'
    assert tag_status(None, None) == 'both_missing'
    assert tag_status('', '') == 'both_missing'


# Selection rules

def test_explicit_selection_wins_over_a_unique_tag_match(tmp_path):
    _register(tmp_path, 'v1', tag='T')
    _register(tmp_path, 'v2', tag='U')
    r = _select(tmp_path, _recording(tmp_path, {'left': 'T'}), explicit='cam-left@v2')
    assert (r['selected'], r['selection_reason'], r['tag_status']) == ('cam-left@v2', 'explicit', 'mismatch')
    assert any('differs' in w for w in r['warnings'])          # advisory, the record is still returned


def test_a_unique_tag_match_is_used_and_recorded(tmp_path):
    _register(tmp_path, 'v1', tag='T')
    _register(tmp_path, 'v2', tag='U')
    _register(tmp_path, 'v3')                                   # untagged candidate
    r = _select(tmp_path, _recording(tmp_path, {'left': 'U'}))
    assert (r['selected'], r['selection_reason'], r['tag_status']) == ('cam-left@v2', 'unique_tag_match', 'match')
    assert r['warnings'] == []
    assert sorted(c['ref'] for c in r['candidates']) == ['cam-left@v1', 'cam-left@v2', 'cam-left@v3']
    assert [c['ref'] for c in r['candidates'] if c['tag_matches']] == ['cam-left@v2']
    for k in ('calibration_digest', 'content_digest', 'replay_paths', 'recording_tag', 'calibration_tag'):
        assert r[k]


@pytest.mark.parametrize('rec_tag,expect', [(MISSING, 'both_missing'), (None, 'both_missing')])
def test_an_untagged_recording_needs_a_default_and_untagged_never_matches_untagged(tmp_path, rec_tag, expect):
    _register(tmp_path, 'v1')                                   # untagged calibration
    rec = _recording(tmp_path, {'left': rec_tag})
    with pytest.raises(SelectionError, match='--calibration'):
        _select(tmp_path, rec)
    r = _select(tmp_path, rec, default='cam-left@v1')
    assert (r['selection_reason'], r['tag_status']) == ('default', expect)
    assert 'states no left hardware tag' in r['selection_detail']


def test_recording_untagged_against_a_tagged_default(tmp_path):
    _register(tmp_path, 'v1', tag='T')
    r = _select(tmp_path, _recording(tmp_path, {'left': MISSING}), default='cam-left@v1')
    assert r['tag_status'] == 'recording_missing'


def test_a_mismatch_falls_back_to_the_default_and_warns(tmp_path):
    _register(tmp_path, 'v1', tag='A')
    rec = _recording(tmp_path, {'left': 'B'})
    with pytest.raises(SelectionError, match="no left calibration is tagged 'B'"):
        _select(tmp_path, rec)
    r = _select(tmp_path, rec, default='cam-left@v1')
    assert (r['selection_reason'], r['tag_status']) == ('default', 'mismatch')
    assert len(r['warnings']) == 2


def test_two_revisions_sharing_a_tag_are_ambiguous_and_use_the_default(tmp_path):
    _register(tmp_path, 'v1', tag='T')
    _register(tmp_path, 'v2', tag='T')
    rec = _recording(tmp_path, {'left': 'T'})
    with pytest.raises(SelectionError, match='2 left calibrations share'):
        _select(tmp_path, rec)
    r = _select(tmp_path, rec, default='cam-left@v2')
    assert (r['selected'], r['selection_reason'], r['tag_status']) == ('cam-left@v2', 'default', 'match')
    assert 'ambiguous' not in r['selection_reason'] and 'share tag' in r['selection_detail']


def test_a_wrong_head_candidate_is_excluded_and_cannot_be_forced(tmp_path):
    _register(tmp_path, 'v1', head='right', tag='T')
    _register(tmp_path, 'v1', head='left')
    rec = _recording(tmp_path, {'left': 'T', 'right': 'T'})
    r = _select(tmp_path, rec, default='cam-left@v1')
    assert r['selected'] == 'cam-left@v1' and r['selection_reason'] == 'default'
    assert [e['ref'] for e in r['excluded']] == ['cam-right@v1']
    with pytest.raises(AssetError, match="for head 'right'"):
        _select(tmp_path, rec, explicit='cam-right@v1')
    with pytest.raises(AssetError, match="for head 'right'"):
        _select(tmp_path, _recording(tmp_path, {'left': None}, 'rec2'), default='cam-right@v1')


def test_corrupt_or_missing_files_keep_their_technical_errors_whatever_the_tags(tmp_path):
    _register(tmp_path, 'v1', tag='T')
    rec = _recording(tmp_path, {'left': 'T'})
    asset = load_asset('cam-left@v1', manifest_dir=tmp_path / 'manifests', store=tmp_path / 'store')
    p = asset.path('calibration/rays.npy')
    p.write_bytes(p.read_bytes()[:-1] + b'\x00')
    with pytest.raises(AssetError, match='not usable'):
        _select(tmp_path, rec)                                   # a perfect tag match does not help
    p.unlink()
    with pytest.raises(AssetError, match='missing'):
        _select(tmp_path, rec, explicit='cam-left@v1')
