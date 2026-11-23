"""
Choose the calibration asset for one head of one recording, and say why.

One callable (`select_calibration`) serves the CLI (`python -m rhabdoforge.hardware select`) and
the replay path. Inputs: the recording directory and head, an optional explicit asset@version and an
optional nominated default. Only assets registered for the requested head are candidates.

    1. An explicit calibration always wins (selection_reason 'explicit').
    2. Otherwise, exactly one candidate whose non-empty hardware_tag equals the recording's
       non-empty tag is used ('unique_tag_match').
    3. Otherwise the nominated default is used ('default'), with the reason no unique match
       existed ('recording_untagged', 'no_tag_match', 'ambiguous_tag_match').
    4. Otherwise selection fails with SelectionError naming the candidates and how to choose.

selection_reason (how the asset was chosen) is distinct from tag_status (how the recording's and
the selected calibration's tags compare): 'match', 'recording_missing', 'calibration_missing',
'both_missing' or 'mismatch'. A missing value never counts as a match. Tag disagreement is
advisory: it is reported in `warnings` and never stops the run. Missing or corrupt asset files,
an explicit/default asset for another head, or an unreadable recording journal are technical
errors (AssetError / SelectionError), unchanged by any tag rule.

The recording journal is read directly (recording.json): no TrilobiteVision import.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .assets import MANIFEST_DIR, AssetError, list_assets, load_asset

SCHEMA = 1


class SelectionError(AssetError):
    """No calibration could be selected (no explicit choice, no unique match, no default)."""


def recording_hardware_tag(recording: Path | str, head: str) -> dict[str, Any]:
    """The tag a TV recording journal states for one head (start snapshot first, then arm)."""
    jpath = Path(recording) / 'recording.json'
    try:
        journal = json.loads(jpath.read_text(encoding='utf-8'))
    except (OSError, ValueError) as e:
        raise SelectionError(f'cannot read the recording journal {jpath}: {e}') from e
    for section in ('geometry_at_start', 'geometry'):
        h = ((journal.get(section) or {}).get('heads') or {}).get(head)
        if isinstance(h, dict) and 'hardware_tag' in h:
            tag = h['hardware_tag'].strip() if isinstance(h['hardware_tag'], str) else ''
            return {'tag_status': 'tagged' if tag else 'unset', 'hardware_tag': tag or None, 'section': section}
    known = set()
    for section in ('geometry_at_start', 'geometry'):
        known |= set(((journal.get(section) or {}).get('heads') or {}).keys())
    if known and head not in known:
        raise SelectionError(f'recording {Path(recording).name} has no head {head!r} (heads: {sorted(known)})')
    return {'tag_status': 'absent', 'hardware_tag': None, 'section': None}


def tag_status(recording_tag: Optional[str], calibration_tag: Optional[str]) -> str:
    if recording_tag and calibration_tag:
        return 'match' if recording_tag == calibration_tag else 'mismatch'
    if not recording_tag and not calibration_tag:
        return 'both_missing'
    return 'recording_missing' if not recording_tag else 'calibration_missing'


_TAG_WARNING = {
    'mismatch': 'recording tag {r!r} differs from calibration tag {c!r}',
    'recording_missing': 'the recording states no hardware tag (calibration tag {c!r})',
    'calibration_missing': 'the calibration states no hardware tag (recording tag {r!r})',
    'both_missing': 'neither the recording nor the calibration states a hardware tag',
}


def _load_for_head(ref: str, head: str, role: str, manifest_dir, store):
    asset = load_asset(ref, manifest_dir=manifest_dir, store=store)
    if asset.manifest.get('head') != head:
        raise AssetError(f'{role} calibration {asset.ref} is for head {asset.manifest.get("head")!r}, '
                         f'not {head!r}')
    return asset


def select_calibration(recording: Path | str, head: str, explicit: Optional[str] = None,
                       default: Optional[str] = None, manifest_dir: Path = MANIFEST_DIR,
                       store: Optional[Path] = None, verify_hashes: bool = True) -> dict[str, Any]:
    """Selection record for (recording, head). See the module docstring for the rules."""
    rec = recording_hardware_tag(recording, head)
    rtag = rec['hardware_tag']

    candidates, excluded = [], []
    for ref in list_assets(manifest_dir):
        a = load_asset(ref, manifest_dir=manifest_dir, store=store)
        row = {'ref': a.ref, 'head': a.manifest.get('head'), 'hardware_tag': a.hardware_tag}
        if row['head'] == head:
            row['tag_matches'] = bool(rtag) and a.hardware_tag == rtag
            candidates.append(row)
        else:
            excluded.append({**row, 'reason': f'head {row["head"]!r} is not {head!r}'})
    matches = [c['ref'] for c in candidates if c['tag_matches']]

    warnings: list[str] = []
    if explicit:
        asset = _load_for_head(explicit, head, 'explicit', manifest_dir, store)
        reason, detail = 'explicit', f'explicitly selected {asset.ref}'
    elif len(matches) == 1:
        asset = load_asset(matches[0], manifest_dir=manifest_dir, store=store)
        reason, detail = 'unique_tag_match', f'the only {head} calibration tagged {rtag!r}'
    else:
        why = ('recording_untagged' if not rtag else
               'ambiguous_tag_match' if len(matches) > 1 else 'no_tag_match')
        explain = {'recording_untagged': f'the recording states no {head} hardware tag',
                   'ambiguous_tag_match': f'{len(matches)} {head} calibrations share tag {rtag!r}: {matches}',
                   'no_tag_match': f'no {head} calibration is tagged {rtag!r}'}[why]
        if not default:
            refs = [c['ref'] for c in candidates]
            raise SelectionError(f'{explain}; no default nominated. Choose one with --calibration '
                                 f'<asset@version> (or nominate --default <asset@version>). '
                                 f'{head} candidates: {refs or "none registered"}')
        asset = _load_for_head(default, head, 'default', manifest_dir, store)
        reason, detail = 'default', f'{explain}; using the nominated default {asset.ref}'
        warnings.append(f'{explain}; fell back to the nominated default {asset.ref}')

    asset.require(hashes=verify_hashes)          # technical integrity: unaffected by tag rules

    status = tag_status(rtag, asset.hardware_tag)
    if status != 'match':
        warnings.append(_TAG_WARNING[status].format(r=rtag, c=asset.hardware_tag))

    return {
        'schema': SCHEMA,
        'decided_at': datetime.now().isoformat(timespec='seconds'),
        'recording': str(Path(recording)),
        'head': head,
        'recording_tag': rtag,
        'recording_tag_source': rec['tag_status'],
        'selected': asset.ref,
        'selection_reason': reason,
        'selection_detail': detail,
        'tag_status': status,
        'calibration_tag': asset.hardware_tag,
        'calibration_tag_assignment': asset.manifest.get('hardware_tag_assignment'),
        'calibration_status': asset.manifest.get('status'),
        'calibration_digest': asset.calibration_digest,
        'content_digest': asset.content_digest,
        'explicit': explicit,
        'default': default,
        'candidates': candidates,
        'excluded': excluded,
        'warnings': warnings,
        'replay_paths': asset.replay_paths(),
    }


def summary_lines(record: dict[str, Any]) -> list[str]:
    lines = [
        f"head {record['head']}: {record['selected']}  ({record['selection_reason']})",
        f"  {record['selection_detail']}",
        f"  tags: recording={record['recording_tag']!r} calibration={record['calibration_tag']!r} "
        f"-> {record['tag_status']}",
        f"  calibration digest {record['calibration_digest'][:16]}..  content {record['content_digest'][:16]}..",
        f"  candidates: {[c['ref'] for c in record['candidates']]}",
    ]
    lines += [f"  warning: {w}" for w in record['warnings']]
    return lines
