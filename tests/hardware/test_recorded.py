"""
Recorded light into the RF receptor model: the incident seam, the light map, the missing-input
policy, structural checks, and a short real replay (skips without the asset, recording or TV).
"""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from rhabdoforge.hardware.recorded import (
    IMPUTED_HOLD, IMPUTED_NONE, IMPUTED_REFERENCE, LightMap, ReceptorMap, apply_missing_policy,
)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'examples'))
import trilobite_eye  # noqa: E402

SCAFFOLD = ROOT / 'assets' / 'trilobite_left_scaffold.npz'
REF = 'trilobite-left@left-room-20261105-scaled004'


# Light map and missing-input policy (pure)

def test_light_map_is_fixed_linear_and_never_negative():
    lm = LightMap(2000.0, 'explicit counts')
    assert np.allclose(lm.drive([0, 1000, 2000, 4000]), [0, 0.5, 1.0, 2.0])
    assert np.allclose(lm.drive([-150.0]), [0.0])                    # noisy negative counts drive 0
    assert lm.describe()['radiance_clamp'] == 100.0
    with pytest.raises(ValueError):
        LightMap(0.0, 'bad')


def test_missing_input_holds_last_valid_and_first_invalid_gets_the_reference():
    last = np.array([np.nan, np.nan, np.nan])
    d, imp, last = apply_missing_policy([0.5, 0.7, 0.9], [True, True, False], last)
    assert np.allclose(d, [0.5, 0.7, 1.0]) and list(imp) == [IMPUTED_NONE, IMPUTED_NONE, IMPUTED_REFERENCE]
    d, imp, last = apply_missing_policy([0.0, 0.8, 0.4], [False, True, True], last)
    assert np.allclose(d, [0.5, 0.8, 0.4]) and list(imp) == [IMPUTED_HOLD, IMPUTED_NONE, IMPUTED_NONE]
    d, imp, _ = apply_missing_policy([0.0, 0.0, 0.0], [False, False, False], last)
    assert np.allclose(d, [0.5, 0.8, 0.4]) and (imp == IMPUTED_HOLD).all()
    assert not np.any(d == 0.0)                                     # missing never reads as darkness


# Structural checks

def _meta_from_scaffold():
    from rhabdoforge.hardware import load_asset
    a = load_asset(REF)
    if not a.verify(roles=('geometry',), hashes=False)['ok']:
        pytest.skip('calibration asset not imported')
    return a.rays_meta()


def test_receptor_map_refuses_a_scaffold_from_another_calibration_and_nonzero_tilt():
    meta = _meta_from_scaffold()
    model = trilobite_eye.build_model('open7', scaffold=str(SCAFFOLD))
    rmap = ReceptorMap(model, SCAFFOLD, meta)
    assert len(rmap) == 130 * 7 and not rmap.fused
    assert set(rmap.group_slot) == set(range(7))
    other = copy.deepcopy(meta)
    other['lenslets'][0]['viewpoint_mm'][0] += 1.0
    with pytest.raises(ValueError, match='not built from this calibration'):
        ReceptorMap(model, SCAFFOLD, other)
    model.interommatidial_tilt = 0.1
    with pytest.raises(NotImplementedError, match='tilt'):
        ReceptorMap(model, SCAFFOLD, meta)


# GPU: the incident seam reproduces the native reduction

@pytest.fixture(scope='module')
def gl():
    try:
        from rhabdoforge.engine import Context
        ctx = Context()
    except Exception as e:                                          # pragma: no cover
        pytest.skip(f'no OpenGL context: {e}')
    yield ctx
    ctx.free()


def _crate_scene():
    from rhabdoforge.engine import Asset, Scene
    from rhabdoforge.engine.meshes import CUBE_INDICES, CUBE_VERTICES
    scene = Scene(background_color=(0.0, 0.0, 0.0))
    scene.add_sky('assets/textures/kloppenheim_05_4k.exr')
    pos, uvs = np.split(CUBE_VERTICES.reshape(-1, 5), [3], axis=1)
    crate = Asset.from_arrays(name='crate', vertices=pos, faces=CUBE_INDICES.reshape(-1, 3), uv_coords=uvs,
                              texture='assets/textures/wood.jpg', sRGB=False)
    scene.add_instance(asset=crate, transform=(float(pos.max(0)[0]), 0.0, -0.7 - float(pos.max(0)[0])))
    return scene


@pytest.mark.gl
@pytest.mark.parametrize('bundle', ['single', 'open7'])
def test_incident_light_equal_to_the_dispatched_light_gives_the_native_outputs(gl, bundle):
    from rhabdoforge.engine import Agent
    from rhabdoforge.renderers import Renderer

    scene = _crate_scene()
    kw = dict(scene=scene, agent=Agent(position=(0, 0, 0)), nb_samples=64, time_dithering=False,
              batch_size=1, context=gl)
    native = Renderer(model=trilobite_eye.build_model(bundle, scaffold=str(SCAFFOLD)), **kw)
    seam = Renderer(model=trilobite_eye.build_model(bundle, scaffold=str(SCAFFOLD)), **kw)
    S = native._samples_per_rhab
    for k in range(4):
        native.step(dt=1 / 60, readback=False)
        rays = native.eye_buffers['rays_intermediate'].read().reshape(native.incident_groups, S, 4)
        w = rays[..., 3:4]
        light = (np.minimum(rays[..., :3], 100.0) * w).sum(1) / np.maximum(w.sum(1), 1e-30)
        light[w.sum(1)[:, 0] <= 0] = 0.0
        seam.step(dt=1 / 60, readback=False, incident=light)
        a = native.eye_buffers['colors'].read(0, native.model.size)
        b = seam.eye_buffers['colors'].read(0, seam.model.size)
        assert np.allclose(a, b, rtol=1e-5, atol=1e-6), f'step {k}'
        da, db = native.eye_buffers['rhab_dynamic'].read(), seam.eye_buffers['rhab_dynamic'].read()
        assert np.allclose(da['curr_acc_angles'], db['curr_acc_angles'])
        assert np.allclose(da['curr_adaptation'], db['curr_adaptation'], rtol=1e-5)
    assert np.ptp(a[:, 3]) > 0 or np.ptp(a[:, :3]) > 0              # a non-trivial scene


@pytest.mark.gl
def test_incident_shape_and_finiteness_are_checked(gl):
    from rhabdoforge.engine import Agent, Scene
    from rhabdoforge.renderers import Renderer
    r = Renderer(model=trilobite_eye.build_model('single', scaffold=str(SCAFFOLD)), scene=Scene(),
                 agent=Agent(position=(0, 0, 0)), nb_samples=1, time_dithering=False, context=gl)
    with pytest.raises(ValueError, match='shape'):
        r.write_incident(np.ones((129, 3)))
    with pytest.raises(ValueError):
        r.write_incident(np.full((130, 3), np.nan))


# Real recording (left-room), a few exposures

def _tv_or_skip():
    try:
        import trilobite
    except ImportError:
        pytest.skip('TrilobiteVision not installed in this environment')
    tv = Path(trilobite.__file__).resolve().parents[2]
    cfg_path = tv / 'config/fieldsample/left-room.yaml'
    if not cfg_path.exists():
        pytest.skip('left-room config absent')
    import yaml
    cfg = yaml.safe_load(cfg_path.read_text())
    if not (tv / cfg['recording'] / 'recording.json').exists():
        pytest.skip('room recording absent')
    _meta_from_scaffold()
    return tv, cfg


def _selection(tv, cfg):
    from rhabdoforge.hardware.selection import select_calibration
    return select_calibration(tv / cfg['recording'], 'left', explicit=REF, verify_hashes=False)


@pytest.mark.gl
def test_a_short_real_replay_matches_tv_sampling_and_records_everything(gl, tmp_path, monkeypatch):
    from rhabdoforge.hardware.recorded import RunOptions, run_recorded
    tv, cfg = _tv_or_skip()
    monkeypatch.chdir(tv)
    cfg = {**cfg, 'frames': {'start': 12500, 'end': 12502}}
    run = run_recorded(cfg, _selection(tv, cfg), tmp_path, RunOptions(light_reference_counts=3000.0), context=gl)
    assert run['exposures']['count'] == 3 and run['totals']['samples'] == 3 * 130
    assert run['calibration_selection']['tag_status'] == 'both_missing'
    assert run['fit_check']['verdicts'] == {'pass': 130}
    A = np.load(tmp_path / 'arrays.npz')
    g = list(A['group_ids']).index('L+00+00/r0')

    from trilobite.fieldsample.replay import ReplaySession, apply_selection
    from trilobite.fieldsample.request import from_dict
    s = ReplaySession(apply_selection(cfg, _selection(tv, cfg)))
    req = from_dict({'request_id': 'c', 'lenslet_id': 'L+00+00',
                     'reference_direction': s.viewports['L+00+00'].reference.tolist(),
                     'offsets_deg': [0, 0], 'fwhm_deg': 1.0})
    for k, eid in enumerate(A['exposure_ids']):
        direct = s.sample(s.read(int(eid)), [req])[0]['value']
        assert A['value'][k, g] == pytest.approx(direct, rel=1e-6)
    assert np.allclose(A['ema_w'], A['drive'], atol=1e-5)          # the seam passes the light unchanged
    assert np.allclose(A['drive'], A['value'] / 3000.0)
    lines = (tmp_path / 'samples.jsonl').read_text().splitlines()
    assert len(lines) == 390 and json.loads(lines[0])['input'] == 'measured'


@pytest.mark.gl
def test_invalid_samples_are_held_flagged_and_never_dark_in_a_real_replay(gl, tmp_path, monkeypatch):
    from trilobite.fieldsample import replay as tv_replay
    from rhabdoforge.hardware.recorded import RunOptions, run_recorded
    tv, cfg = _tv_or_skip()
    monkeypatch.chdir(tv)
    cfg = {**cfg, 'frames': {'start': 12500, 'end': 12502}}
    real = tv_replay.ReplaySession.sample

    def flaky(self, frame, requests, keep_weights=False):
        out = real(self, frame, requests, keep_weights)
        eid = frame['exposure']['id']
        for r in out:
            lid = r['lenslet_id']
            if (eid == 12501 and lid == 'L+00+00') or lid == 'L+02-03':
                r.update(validity='invalid', value=None, reason='test: centre outside')
        return out

    monkeypatch.setattr(tv_replay.ReplaySession, 'sample', flaky)
    run = run_recorded(cfg, _selection(tv, cfg), tmp_path,
                       RunOptions(light_reference_counts=3000.0), context=gl)
    A = np.load(tmp_path / 'arrays.npz')
    ids = list(A['group_ids'])
    a, b = ids.index('L+00+00/r0'), ids.index('L+02-03/r0')
    assert list(A['imputed'][:, a]) == [IMPUTED_NONE, IMPUTED_HOLD, IMPUTED_NONE]
    assert A['drive'][1, a] == A['drive'][0, a]                     # held, not zero
    assert (A['imputed'][:, b] == IMPUTED_REFERENCE).all() and np.allclose(A['drive'][:, b], 1.0)
    assert run['totals']['invalid_inputs'] == 4
    assert run['totals']['first_invalid']['policy'] == 'reference_first_invalid'
