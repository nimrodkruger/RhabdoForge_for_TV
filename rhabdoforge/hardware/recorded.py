"""
Recorded light into the RhabdoForge receptor model (TrilobiteVision recordings).

One receptor model, two light sources. Natively, Renderer.step() samples the scene (dispatch) and
reduces it. Here, each exposure of a recording supplies the incident light instead: for every RF ray
group, TrilobiteVision samples the recorded frame through the calibrated lenslet with the field the
receptor has right now, and Renderer.step(incident=...) feeds that light to the unchanged reduction
(sensitivity, pupil, concentration, membrane, gain) and dynamics.

The recording fixes the camera's trajectory and exposures. The model may move its own receptors
(microsaccades) and is then sampled where it looks, inside the recorded field; nothing it does can
steer the camera.

Per exposure, in native order:
    1. read the exposure once (TV ReplaySession.read)
    2. read the receptor state the step will use from the GPU (direction, acceptance widths) and map
       it to the calibrated head frame and lenslet (TV FieldRequest: direction, tangent = RF's T,
       GaussianKernel(FWHM minor along T, major along B))
    3. sample all requests in one batch (TV ReplaySession.sample)
    4. counts -> light with one fixed, saved reference (no per-frame normalisation); negative
       counts drive 0 and the signed value is kept; invalid samples hold the last valid light for
       that receptor and are flagged (never read as darkness)
    5. step RF once with that light; read the outputs synchronously from the GPU (no PBO lag)

Ownership: the driver only reads GPU state and writes the scratch rays buffer. It never writes model
buffers during a run (Renderer.sync_cpu would upload stale CPU dynamic rows over live GPU state).

TrilobiteVision is imported only here (optional dependency of the recorded-input path).
"""
from __future__ import annotations

import json
import platform
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .frames import match_rows, rf_sampling_frame, rf_to_tv_vectors, scaffold_from_rays_meta

IMPUTED_NONE, IMPUTED_HOLD, IMPUTED_REFERENCE = 0, 1, 2
IMPUTED_NAMES = {IMPUTED_NONE: 'measured', IMPUTED_HOLD: 'held_last_valid', IMPUTED_REFERENCE: 'reference_first_invalid'}
RADIANCE_CLAMP = 100.0      # reduction.comp MAX_SAMPLE_RADIANCE


@dataclass
class LightMap:
    """Monochrome counts -> RF relative light, fixed for the whole experiment."""
    counts_reference: float          # dark-subtracted counts that map to light 1.0
    source: str                      # how the reference was obtained
    negative: str = 'clamp_zero'     # negative (noisy) counts drive 0; the signed value is kept

    def __post_init__(self):
        if not (np.isfinite(self.counts_reference) and self.counts_reference > 0):
            raise ValueError('counts_reference must be a positive finite number')

    def drive(self, counts: np.ndarray) -> np.ndarray:
        return np.maximum(np.asarray(counts, dtype=np.float64), 0.0) / self.counts_reference

    def describe(self) -> dict[str, Any]:
        return {**asdict(self), 'light_units': 'relative: 1.0 = counts_reference (RF lum_ref is 1.0 by default)',
                'channels': 'replicated to equal R, G, B', 'radiance_clamp': RADIANCE_CLAMP}


def apply_missing_policy(drive: np.ndarray, valid: np.ndarray, last_valid: np.ndarray,
                         reference: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Model drive for one exposure under the missing-input policy.

    Valid samples keep their measured light. An invalid sample holds that ray group's last valid
    light (IMPUTED_HOLD); with no earlier valid light (first-invalid) it gets the reference light
    (IMPUTED_REFERENCE). Missing input is never read as darkness. Returns (drive, imputed codes,
    updated last_valid); the inputs are not modified.
    """
    drive = np.array(drive, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    last = np.array(last_valid, dtype=np.float64)
    imputed = np.zeros(drive.shape, dtype=np.int8)
    hold = ~valid & np.isfinite(last)
    first = ~valid & ~np.isfinite(last)
    drive[hold], imputed[hold] = last[hold], IMPUTED_HOLD
    drive[first], imputed[first] = reference, IMPUTED_REFERENCE
    last[valid] = drive[valid]
    return drive, imputed, last


class ReceptorMap:
    """RF ray groups -> calibrated lenslets, verified against the selected calibration."""

    def __init__(self, model, scaffold_path: str | Path, rays_meta: dict, scale: float = 1e-6):
        with np.load(scaffold_path) as s:
            scaffold_pos = s['positions']
        expected = scaffold_from_rays_meta(rays_meta)
        if expected['positions'].shape != scaffold_pos.shape or not np.array_equal(expected['positions'], scaffold_pos):
            raise ValueError(f'scaffold {scaffold_path} was not built from this calibration '
                             f'(viewpoints differ): structurally incompatible')
        tilt = np.asarray(model.interommatidial_tilt, dtype=np.float64)
        if np.any(np.abs(tilt) > 1e-9):
            raise NotImplementedError('non-zero ioa_tilt: the RF/TV tilt sign is not GPU-verified yet; '
                                      'use the preset (tilt 0)')
        self.model = model
        self.rows = match_rows(model.positions, scaffold_pos, scale=scale)
        self.omm_lenslet = expected['lenslet_ids'][self.rows]
        N, R = model.shape
        self.fused = bool(model.bundle.fused_rhabdoms)
        # ray group g -> the rhabdomere whose direction dispatch samples around (slot 0 if fused)
        self.group_rhab = np.arange(N) * R if self.fused else np.arange(N * R)
        self.group_omm = self.group_rhab // R
        self.group_slot = self.group_rhab % R
        self.group_lenslet = self.omm_lenslet[self.group_omm]
        self.right = np.asarray(model.buffer['right']).reshape(N, 3)

    def __len__(self) -> int:
        return len(self.group_rhab)

    def group_ids(self) -> list[str]:
        return [f'{l}/r{s}' for l, s in zip(self.group_lenslet, self.group_slot)]


def read_receptor_state(renderer, rmap: ReceptorMap) -> dict[str, np.ndarray]:
    """The live GPU state of each ray group's receptor (what the next dispatch would use)."""
    dyn = renderer.eye_buffers['rhab_dynamic'].read()
    F = np.asarray(dyn['curr_direction'], dtype=np.float64)[rmap.group_rhab]
    acc = np.asarray(dyn['curr_acc_angles'], dtype=np.float64)[rmap.group_rhab]
    T, B = rf_sampling_frame(F, rmap.right[rmap.group_omm])
    return {'direction_rf': F, 'tangent_rf': T, 'fwhm_rad': acc}


def build_requests(state: dict[str, np.ndarray], rmap: ReceptorMap, exposure_id: int):
    from trilobite.fieldsample.kernel import GaussianKernel
    from trilobite.fieldsample.request import FieldRequest

    d_tv = rf_to_tv_vectors(state['direction_rf'])
    t_tv = rf_to_tv_vectors(state['tangent_rf'])
    reqs = []
    for g, gid in enumerate(rmap.group_ids()):
        fa, fb = state['fwhm_rad'][g]
        reqs.append(FieldRequest(f'{gid}@{exposure_id}', str(rmap.group_lenslet[g]),
                                 tuple(float(x) for x in d_tv[g]), GaussianKernel(float(fa), float(fb)),
                                 tangent=tuple(float(x) for x in t_tv[g])))
    return reqs


def fit_check(session, requests, motion: Optional[dict], tolerances=None) -> dict[str, Any]:
    """TV fit check of every receptor's rest field (and its excursion, when motion is given)."""
    from trilobite.fieldsample.fit import check_fit

    rep = check_fit(session.calibration, session.viewports, [(r, motion) for r in requests],
                    tolerances or session.tolerances)
    verdicts: dict[str, int] = {}
    for f in rep['fields']:
        verdicts[f['geometric_verdict']] = verdicts.get(f['geometric_verdict'], 0) + 1
    worst = [f for f in rep['fields'] if f['geometric_verdict'] != 'pass']
    return {'verdicts': verdicts, 'worst': rep['worst'], 'tolerances': rep['tolerances'],
            'optical': sorted({f['criteria']['optical']['verdict'] for f in rep['fields']}),
            'not_passing': [{'request_id': f['field']['request_id'], 'verdict': f['geometric_verdict'],
                             'coverage': f['criteria']['coverage']['verdict'],
                             'sampling': f['criteria']['sampling']['verdict']} for f in worst]}


def _git_rev(path: Path) -> Optional[str]:
    try:
        out = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=path, capture_output=True, text=True)
        dirty = subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=path,
                               capture_output=True, text=True).stdout.strip()
        return out.stdout.strip() + ('+dirty' if dirty else '') if out.returncode == 0 else None
    except OSError:
        return None


@dataclass
class RunOptions:
    bundle: str = 'single'
    fwhm_deg: Any = 1.0
    enable_microsaccades: bool = False
    light_reference_counts: Optional[float] = None
    light_reference_exposure: Optional[int] = None    # median valid rest sample at this exposure
    first_dt_s: Optional[float] = None                # default: median exposure interval
    extra: dict = field(default_factory=dict)


def run_recorded(config: dict[str, Any], selection: dict[str, Any], output: str | Path,
                 opts: RunOptions = RunOptions(), context=None) -> dict[str, Any]:
    """Replay a recording's exposures through the hardware-eye preset with recorded light."""
    import sys
    from trilobite.fieldsample.replay import ReplaySession, apply_selection, selection_summary

    from rhabdoforge.engine import Agent, Context, Scene
    from rhabdoforge.renderers import Renderer

    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / 'examples'))
    import trilobite_eye

    t_start = time.perf_counter()
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)

    cfg = apply_selection(config, selection)
    session = ReplaySession(cfg)
    ids = list(session.selection['ids'])

    own_context = context is None
    ctx = context or Context()
    model = trilobite_eye.build_model(opts.bundle, opts.fwhm_deg, scaffold=str(root / trilobite_eye.SCAFFOLD))
    rmap = ReceptorMap(model, root / trilobite_eye.SCAFFOLD, session.calibration.metadata)
    renderer = Renderer(model=model, scene=Scene(background_color=(0.0, 0.0, 0.0)), agent=Agent(position=(0, 0, 0)),
                        nb_samples=1, time_dithering=False, batch_size=1,
                        enable_microsaccades=opts.enable_microsaccades, context=ctx)
    if renderer.incident_groups != len(rmap):
        raise RuntimeError('ray-group count differs between renderer and receptor map')

    # Fit check of the fields the model will request (rest state, or its excursion when moving)
    rest_state = read_receptor_state(renderer, rmap)
    rest_reqs = build_requests(rest_state, rmap, ids[0])
    motion = None
    if opts.enable_microsaccades:
        r = np.deg2rad(trilobite_eye.MAX_SACCADE_DEG)
        motion = {'type': 'region', 'a_rad': [-r, r], 'b_rad': [-r, r], 'interval_rad': r / 2}
    fit = fit_check(session, rest_reqs, motion)

    # Light reference: explicit counts, or the median valid rest sample of one named exposure
    if opts.light_reference_counts is not None:
        lmap = LightMap(float(opts.light_reference_counts), 'explicit counts')
    else:
        ref_id = opts.light_reference_exposure if opts.light_reference_exposure is not None else ids[0]
        ref = session.sample(session.read(ref_id), build_requests(rest_state, rmap, ref_id))
        vals = [r['value'] for r in ref if r['validity'] != 'invalid' and r['value'] is not None]
        lmap = LightMap(float(np.median(vals)), f'median of {len(vals)} valid rest samples at exposure {ref_id}')

    G = len(rmap)
    E = len(ids)
    rec = {k: np.full((E, G), np.nan) for k in ('value', 'drive', 'valid_mass', 'gain', 'response', 'ema_w')}
    rec['validity'] = np.empty((E, G), dtype=object)
    rec['imputed'] = np.zeros((E, G), dtype=np.int8)
    rec['direction_rf'] = np.zeros((E, G, 3))
    rec['fwhm_rad'] = np.zeros((E, G, 2))
    rec['next_direction_rf'] = np.zeros((E, G, 3))
    times = {'t_mono': [], 't_wall': [], 'dt': [], 'gap_before': [], 'exposure_us': []}

    exposures = [session._by_seq[i] for i in ids]
    t_mono = np.array([float(x['t_mono']) for x in exposures])
    intervals = np.diff(t_mono)
    first_dt = opts.first_dt_s if opts.first_dt_s is not None else (float(np.median(intervals)) if len(intervals) else 0.2)

    last_valid = np.full(G, np.nan)
    samples_rows = []
    for k, eid in enumerate(ids):
        frame = session.read(eid)                                    # 1
        state = read_receptor_state(renderer, rmap)                  # 2 (state this step uses)
        reqs = build_requests(state, rmap, eid)
        results = session.sample(frame, reqs)                        # 3 (one batch)
        value = np.array([np.nan if r['value'] is None else r['value'] for r in results], dtype=np.float64)
        valid = np.array([r['validity'] != 'invalid' and r['value'] is not None for r in results])
        drive, imputed, last_valid = apply_missing_policy(lmap.drive(np.where(valid, value, 0.0)), valid, last_valid)  # 4

        dt = first_dt if k == 0 else float(t_mono[k] - t_mono[k - 1])
        light = np.repeat(drive[:, None], 3, axis=1).astype(np.float32)
        renderer.step(dt=dt, readback=False, incident=light)         # 5
        colors = renderer.eye_buffers['colors'].read(0, model.size)
        ema = renderer.eye_buffers['ema_state'].read()
        nxt = read_receptor_state(renderer, rmap)

        g_r = rmap.group_rhab
        rec['value'][k], rec['drive'][k], rec['imputed'][k] = value, drive, imputed
        rec['validity'][k] = [r['validity'] for r in results]
        rec['valid_mass'][k] = [np.nan if r.get('valid_mass_fraction') is None else r['valid_mass_fraction'] for r in results]
        rec['gain'][k] = colors[g_r, 3]
        rec['response'][k] = colors[g_r, :3].mean(axis=1)
        rec['ema_w'][k] = ema[g_r, 3]
        rec['direction_rf'][k], rec['fwhm_rad'][k] = state['direction_rf'], state['fwhm_rad']
        rec['next_direction_rf'][k] = nxt['direction_rf']
        ex = frame['exposure']
        times['t_mono'].append(ex.get('t_mono')), times['t_wall'].append(ex.get('t_wall'))
        times['dt'].append(dt), times['gap_before'].append(ex.get('gap_before')), times['exposure_us'].append(ex.get('exposure_us'))
        for g, r in enumerate(results):
            samples_rows.append({'exposure_id': eid, 'group': rmap.group_ids()[g], 'lenslet_id': str(rmap.group_lenslet[g]),
                                 'value': r['value'], 'validity': r['validity'], 'reason': r.get('reason'),
                                 'drive': float(drive[g]), 'input': IMPUTED_NAMES[int(imputed[g])],
                                 'response': float(rec['response'][k, g]), 'gain': float(rec['gain'][k, g])})

    invalid = int(np.sum(rec['imputed'] > 0))
    first_invalid = None
    if invalid:
        e, g = np.argwhere(rec['imputed'] > 0)[0]
        first_invalid = {'exposure_id': int(ids[e]), 'group': rmap.group_ids()[g],
                         'policy': IMPUTED_NAMES[int(rec['imputed'][e, g])]}
    clamped = int(np.sum(rec['drive'] > RADIANCE_CLAMP))

    run = {
        'schema': 1,
        'kind': 'recorded light through the RF hardware-eye preset',
        'calibration_selection': selection_summary(cfg),
        'recording': cfg['recording'], 'head': cfg['head'],
        'exposures': {'ids': ids, 'count': E, 'first_dt_s': first_dt, 'journal_gaps': session.selection.get('journal_gaps_in_span'),
                      'timestamp_convention': exposures[0].get('timestamp_convention') if exposures else None,
                      'note': 'one RF step per exposure, held stimulus, dt = t_mono difference; '
                              f'{1 / first_dt:.1f} Hz cannot resolve millisecond physiology'},
        'model': {'preset': 'examples/trilobite_eye.py', 'bundle': opts.bundle, 'fwhm_deg': opts.fwhm_deg,
                  'shape': list(model.shape), 'ray_groups': G, 'fused': rmap.fused,
                  'enable_microsaccades': opts.enable_microsaccades},
        'renderer': {'lum_ref': renderer.reference_luminance, 'noise_threshold': renderer.noise_threshold,
                     'photon_concentration': renderer.photon_concentration, 'nb_samples': 1,
                     'readback': 'synchronous colors/ema/rhab_dynamic reads after each step (no PBO lag)',
                     'gain_timing': 'the response at exposure k uses the adaptation gain computed by the '
                                    'dynamics of step k-1 (native order: dispatch/incident -> reduction -> dynamics)'},
        'light_map': lmap.describe(),
        'missing_input_policy': {'hold': 'invalid sample -> last valid light of that ray group, flagged',
                                 'first_invalid': 'no earlier valid light -> reference light 1.0, flagged',
                                 'never': 'missing input is never treated as measured darkness'},
        'fit_check': fit,
        'totals': {'samples': E * G, 'invalid_inputs': invalid, 'first_invalid': first_invalid,
                   'drive_clamped_by_rf': clamped, 'negative_counts': int(np.sum(rec['value'] < 0))},
        'tv_identity': session.identity(),
        'versions': {'rhabdoforge': _git_rev(root), 'trilobitevision': None,
                     'python': platform.python_version(), 'numpy': np.__version__},
        'seconds': round(time.perf_counter() - t_start, 2),
    }
    try:
        import trilobite
        run['versions']['trilobitevision'] = _git_rev(Path(trilobite.__file__).resolve().parents[2])
    except Exception:
        pass

    np.savez_compressed(out / 'arrays.npz', exposure_ids=np.array(ids), group_ids=np.array(rmap.group_ids()),
                        lenslet_ids=np.array(rmap.group_lenslet), t_mono=np.array(times['t_mono'], float),
                        dt=np.array(times['dt']), **{k: v for k, v in rec.items() if k != 'validity'},
                        validity=rec['validity'].astype(str))
    with open(out / 'samples.jsonl', 'w', encoding='utf-8') as f:
        for row in samples_rows:
            f.write(json.dumps(row) + '\n')
    (out / 'run.json').write_text(json.dumps(run, indent=1, default=_json_default), encoding='utf-8')
    if own_context:
        ctx.free()
    return run


def _json_default(v):
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, Path):
        return str(v)
    return str(v)
