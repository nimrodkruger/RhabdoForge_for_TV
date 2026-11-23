"""
Named, versioned calibration assets for physical eyes (hardware), resolved outside the source tree.

A small manifest is tracked in the package (rhabdoforge/hardware/manifests/<asset>@<version>.json).
It names every file the asset needs, with its role, size and SHA-256, and where it came from.
The heavy arrays live in a local store, never in package source:

    <store>/<asset>/<version>/<path listed in the manifest>

The store is $RHABDOFORGE_HARDWARE_ROOT, or the platform user-data folder
(%LOCALAPPDATA%/rhabdoforge/hardware on Windows, ~/.local/share/rhabdoforge/hardware elsewhere).
A second machine reads the manifest to know exactly which files to fetch, then `import`s them
from a TrilobiteVision checkout (or any copy keeping the manifest's `source` paths) and `verify`s.

Lookup is by explicit reference "asset@version". A bare "asset" resolves only when exactly one
version is registered; otherwise it is an error listing the versions (no silent pick).

Roles:
    geometry        calibration bundle: rays.json, rays.npy, lenslet.npy (TV ray export)
    photometry      per-session photometric references the replay needs: dark mean, pixel std,
                    hot pixels, their descriptor, the native ROI and the calibration run record
    source_record   the journal (recording.json) of the recording the calibration came from, no frames;
                    TV checks a replayed recording's geometry_at_start against it

The photometry and source_record lists mirror the files TrilobiteVision's ReplaySession reads
(audited 2026-11-23, left-room config; TV opens the source journal with require_chunks=False since
that date). A change on the TV side needs a new asset version.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Optional

MANIFEST_DIR = Path(__file__).resolve().parent / 'manifests'
SCHEMA = 1
ROLES = ('geometry', 'photometry', 'source_record')

GEOMETRY_FILES = ('rays.json', 'rays.npy', 'lenslet.npy')
REFERENCE_FILES = ('descriptor.json', 'dark_mean.npy', 'pix_std.npy', 'hot.npy')


class AssetError(RuntimeError):
    """Missing, ambiguous, corrupt or structurally incompatible hardware asset (a technical error)."""


def store_root() -> Path:
    env = os.environ.get('RHABDOFORGE_HARDWARE_ROOT')
    if env:
        return Path(env).expanduser()
    if sys.platform.startswith('win'):
        base = Path(os.environ.get('LOCALAPPDATA', Path.home() / 'AppData' / 'Local'))
    else:
        base = Path(os.environ.get('XDG_DATA_HOME', Path.home() / '.local' / 'share'))
    return base / 'rhabdoforge' / 'hardware'


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def digest_of(entries: list[dict]) -> str:
    """Content digest of a set of files: SHA-256 of sorted 'path sha256' lines."""
    lines = sorted(f"{e['path']} {e['sha256']}" for e in entries)
    return hashlib.sha256('\n'.join(lines).encode()).hexdigest()


def list_assets(manifest_dir: Path = MANIFEST_DIR) -> list[str]:
    return sorted(p.stem for p in Path(manifest_dir).glob('*@*.json'))


@dataclass
class HardwareAsset:
    manifest: dict[str, Any]
    manifest_path: Path
    store: Path

    # Identity

    @property
    def ref(self) -> str:
        return f"{self.manifest['asset']}@{self.manifest['version']}"

    @property
    def hardware_tag(self) -> Optional[str]:
        return self.manifest.get('hardware_tag')

    @property
    def calibration_digest(self) -> str:
        return self.manifest['calibration_digest']

    @property
    def content_digest(self) -> str:
        return self.manifest['content_digest']

    # Paths

    @property
    def root(self) -> Path:
        return self.store / self.manifest['asset'] / self.manifest['version']

    def path(self, rel: str) -> Path:
        return self.root / rel

    def files(self, role: Optional[str] = None) -> list[dict]:
        return [e for e in self.manifest['files'] if role is None or e['role'] == role]

    @property
    def geometry_dir(self) -> Path:
        return self.path(self.manifest['layout']['geometry'])

    @property
    def rays_json(self) -> Path:
        return self.geometry_dir / 'rays.json'

    def replay_paths(self) -> dict[str, str]:
        """Paths for a TrilobiteVision ReplaySession config, all inside the local store."""
        lay = self.manifest['layout']
        pho = self.manifest['photometry']
        return {
            'bundle': str(self.path(lay['geometry'])),
            'calibration_run': str(self.path(lay['calibration_run'])),
            'calibration_source_recording': str(self.path(lay['source_recording'])),
            'session_id': pho['session_id'],
            'head': self.manifest['head'],
            'preprocessing': pho['preprocessing'],
        }

    def rays_meta(self) -> dict:
        self.require('geometry')
        return json.loads(self.rays_json.read_text())

    # Integrity

    def verify(self, roles: Optional[tuple[str, ...]] = None, hashes: bool = True) -> dict[str, Any]:
        """Check every listed file exists with its size (and SHA-256). Never repairs anything."""
        problems = []
        for e in self.manifest['files']:
            if roles and e['role'] not in roles:
                continue
            p = self.path(e['path'])
            if not p.is_file():
                problems.append({'path': e['path'], 'problem': 'missing'})
                continue
            if p.stat().st_size != e['bytes']:
                problems.append({'path': e['path'], 'problem': 'size', 'expected': e['bytes'],
                                 'found': p.stat().st_size})
                continue
            if hashes and sha256_file(p) != e['sha256']:
                problems.append({'path': e['path'], 'problem': 'sha256'})
        return {'asset': self.ref, 'root': str(self.root), 'ok': not problems, 'problems': problems,
                'checked': 'sha256' if hashes else 'size'}

    def require(self, *roles: str, hashes: bool = False) -> None:
        report = self.verify(roles=roles or None, hashes=hashes)
        if not report['ok']:
            raise AssetError(f"{self.ref} is not usable from {self.root}: {report['problems']}. "
                             f"Run: python -m rhabdoforge.hardware import {self.ref} --from <TrilobiteVision root>")

    # Import

    def import_from(self, source_root: Path, overwrite: bool = False) -> dict[str, Any]:
        """Copy each listed file from `source_root / entry['source']` into the store, then verify."""
        source_root = Path(source_root)
        for e in self.manifest['files']:
            src = source_root / e['source']
            dst = self.path(e['path'])
            if not src.is_file():
                raise AssetError(f'source file missing: {src}')
            if sha256_file(src) != e['sha256']:
                raise AssetError(f'source file differs from the manifest: {src}')
            if dst.exists() and not overwrite and dst.stat().st_size == e['bytes'] and sha256_file(dst) == e['sha256']:
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        return self.verify()


def load_asset(ref: str, manifest_dir: Path = MANIFEST_DIR, store: Optional[Path] = None) -> HardwareAsset:
    """Resolve 'asset@version' (or a bare 'asset' with exactly one version) to its manifest."""
    known = list_assets(manifest_dir)
    if '@' in ref:
        if ref not in known:
            raise AssetError(f'unknown hardware asset {ref!r}; registered: {known}')
        name = ref
    else:
        matches = [k for k in known if k.split('@', 1)[0] == ref]
        if len(matches) != 1:
            raise AssetError(f'{ref!r} matches {len(matches)} registered versions {matches}; '
                             f'name one explicitly as asset@version')
        name = matches[0]
    path = Path(manifest_dir) / f'{name}.json'
    manifest = json.loads(path.read_text())
    if manifest.get('schema') != SCHEMA:
        raise AssetError(f'{path.name}: unsupported manifest schema {manifest.get("schema")}')
    return HardwareAsset(manifest=manifest, manifest_path=path, store=Path(store) if store else store_root())


# Registration (developer side, from a TrilobiteVision checkout)

def build_manifest(*, asset: str, version: str, label: str, status: str, source_root: Path,
                   bundle: str, calibration_run: str, session_id: str, head: str,
                   source_recording: str, provenance: dict[str, Any],
                   hardware_tag: Optional[str] = None, tag_assignment: Optional[dict] = None,
                   preprocessing: str = 'dark_only') -> dict[str, Any]:
    """Manifest for a TV calibration: lists, sizes and hashes the files ReplaySession needs."""
    source_root = Path(source_root)
    entries: list[dict] = []

    def add(role: str, dst: str, src: str) -> None:
        p = source_root / src
        if not p.is_file():
            raise AssetError(f'missing source file: {p}')
        entries.append({'path': dst, 'role': role, 'bytes': p.stat().st_size,
                        'sha256': sha256_file(p), 'source': Path(src).as_posix()})

    for name in GEOMETRY_FILES:
        add('geometry', f'calibration/{name}', f'{bundle}/{name}')

    add('photometry', 'run/run.json', f'{calibration_run}/run.json')
    for name in REFERENCE_FILES:
        add('photometry', f'run/references/{session_id}/{head}/{name}',
            f'{calibration_run}/references/{session_id}/{head}/{name}')
    add('photometry', f'run/lattice/{session_id}/{head}/roi.json',
        f'{calibration_run}/lattice/{session_id}/{head}/roi.json')

    # Only the journal: TV reads geometry_at_start and identity from it, never its frames
    rec_name = Path(source_recording).name
    add('source_record', f'source_recording/{rec_name}/recording.json', f'{source_recording}/recording.json')

    rays = json.loads((source_root / bundle / 'rays.json').read_text())
    geometry = [e for e in entries if e['role'] == 'geometry']
    tag, assignment, tag_warnings = resolve_registration_tag(rays, hardware_tag, tag_assignment)

    return {
        'schema': SCHEMA,
        'asset': asset,
        'version': version,
        'label': label,
        'status': status,
        'head': head,
        'hardware_tag': tag,
        'hardware_tag_assignment': assignment,
        'registration_warnings': tag_warnings,
        'calibration_digest': digest_of(geometry),
        'content_digest': digest_of(entries),
        'calibration': {
            'model': rays['model']['model'], 'trial': rays.get('trial'), 'camera': rays.get('camera'),
            'status': rays.get('status'), 'lenslets': len(rays['lenslets']), 'frame': rays['frame'],
        },
        'rf_frame_transform': {
            'matrix': [[1, 0, 0], [0, -1, 0], [0, 0, -1]],
            'unit_scale': 1000.0,
            'from': 'TV head frame, mm (x native u, y native v, z toward scene)',
            'to': 'RF eye frame, um (+X right, +Y up, -Z forward); examples then scale by 1e-6 to metres',
        },
        'layout': {
            'geometry': 'calibration',
            'calibration_run': 'run',
            'source_recording': f'source_recording/{rec_name}',
        },
        'photometry': {'session_id': session_id, 'preprocessing': preprocessing},
        'files': entries,
        'provenance': {**provenance, 'registered': date.today().isoformat()},
    }


def resolve_registration_tag(rays: dict, explicit_tag: Optional[str],
                             explicit: Optional[dict]) -> tuple[Optional[str], dict, list[str]]:
    """
    Tag of a registered asset: the calibration export's own identity, unless an operator
    states one at registration (explicit wins). A disagreement is kept visible in the
    manifest's registration_warnings and the assignment record, never resolved silently.
    """
    exported = rays.get('hardware_tag')
    exported = exported.strip() if isinstance(exported, str) and exported.strip() else None
    exported_assignment = rays.get('hardware_tag_assignment')
    export_warnings = ((rays.get('provenance') or {}).get('recordings') or {}).get('tag_warnings') or []
    warnings = [f'calibration export: {w}' for w in export_warnings]

    if explicit_tag is not None:
        explicit = dict(explicit or {})
        tag = explicit_tag.strip()
        who = (explicit.get('assigned_by') or '').strip()
        if not tag or not who:
            raise AssetError('an explicit hardware tag needs both the tag and who assigned it')
        if exported and exported != tag:
            warnings.append(f'registration tag {tag!r} differs from the calibration export tag {exported!r}')
        return tag, {'source': 'registration', 'basis': 'explicit', 'assigned_by': who,
                     'note': explicit.get('note', ''), 'date': explicit.get('date') or date.today().isoformat(),
                     'export_tag': exported, 'export_assignment': exported_assignment}, warnings

    if exported:
        return exported, {'source': 'calibration export', **(exported_assignment or {})}, warnings

    if 'hardware_tag' in rays:
        note = 'untagged: the calibration export states no hardware tag'
    else:
        note = 'untagged: no hardware tag was recorded for this calibration; none is asserted'
    return None, {'source': 'none', 'assigned_by': None, 'note': note,
                  'export_assignment': exported_assignment}, warnings


def write_manifest(manifest: dict[str, Any], manifest_dir: Path = MANIFEST_DIR, replace: bool = False) -> Path:
    """Write a manifest; refuses to change the files or tag of an existing asset@version."""
    path = Path(manifest_dir) / f"{manifest['asset']}@{manifest['version']}.json"
    if path.exists() and not replace:
        old = json.loads(path.read_text())
        changed = [k for k in ('content_digest', 'calibration_digest', 'hardware_tag', 'head')
                   if old.get(k) != manifest.get(k)]
        if changed:
            raise AssetError(f'{path.name} is already registered with different {changed}; '
                             f'register a new version (or pass replace=True knowingly)')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=1) + '\n')
    return path
