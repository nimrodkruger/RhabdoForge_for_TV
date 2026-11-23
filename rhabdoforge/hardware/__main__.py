"""
python -m rhabdoforge.hardware list
python -m rhabdoforge.hardware show   ASSET@VERSION
python -m rhabdoforge.hardware where  ASSET@VERSION
python -m rhabdoforge.hardware verify ASSET@VERSION [--quick]
python -m rhabdoforge.hardware import ASSET@VERSION --from <TrilobiteVision root>
python -m rhabdoforge.hardware select --recording <dir> --head left [--calibration A@V] [--default A@V] [--out sel.json]
python -m rhabdoforge.hardware register --asset ... --version ... --from <TV root> --bundle ... (developer)
"""
import argparse
import json
import sys
from pathlib import Path

from .assets import AssetError, build_manifest, list_assets, load_asset, store_root, write_manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog='python -m rhabdoforge.hardware')
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('list')
    for name in ('show', 'where'):
        sub.add_parser(name).add_argument('ref')
    v = sub.add_parser('verify')
    v.add_argument('ref')
    v.add_argument('--quick', action='store_true', help='check presence and sizes only')
    i = sub.add_parser('import')
    i.add_argument('ref')
    i.add_argument('--from', dest='source', required=True, type=Path)
    i.add_argument('--overwrite', action='store_true')

    r = sub.add_parser('register', help='write a manifest from a TrilobiteVision checkout')
    r.add_argument('--asset', required=True)
    r.add_argument('--version', required=True)
    r.add_argument('--label', required=True)
    r.add_argument('--status', required=True)
    r.add_argument('--from', dest='source', required=True, type=Path)
    r.add_argument('--bundle', required=True)
    r.add_argument('--calibration-run', required=True)
    r.add_argument('--session-id', required=True)
    r.add_argument('--head', required=True)
    r.add_argument('--source-recording', required=True)
    r.add_argument('--tv-revision', required=True)
    r.add_argument('--tv-config', required=True)
    r.add_argument('--note', default='')
    r.add_argument('--hardware-tag', help="explicit optical-setup identity (overrides the export's)")
    r.add_argument('--tag-assigned-by', help='who states --hardware-tag (required with it)')
    r.add_argument('--tag-note', default='', help='evidence for --hardware-tag')
    r.add_argument('--replace', action='store_true', help='overwrite an existing asset@version')

    sel = sub.add_parser('select', help='choose the calibration for a recording head')
    sel.add_argument('--recording', required=True, type=Path)
    sel.add_argument('--head', required=True)
    sel.add_argument('--calibration', help='explicit asset@version (always wins)')
    sel.add_argument('--default', help='nominated asset@version when no unique tag match exists')
    sel.add_argument('--out', type=Path, help='write the selection record (JSON) here')

    rl = sub.add_parser('replay-light', help='recorded light through the hardware-eye preset (needs TrilobiteVision)')
    rl.add_argument('config', type=Path, help='TrilobiteVision fieldsample config (recording, head, frames)')
    rl.add_argument('--selection', required=True, type=Path, help='record from the select command')
    rl.add_argument('--out', required=True, type=Path)
    rl.add_argument('--bundle', choices=('single', 'open7'), default='single')
    rl.add_argument('--fwhm-deg', type=float, default=1.0)
    rl.add_argument('--microsaccades', action='store_true')
    grp = rl.add_mutually_exclusive_group()
    grp.add_argument('--light-reference-counts', type=float, help='counts mapped to light 1.0')
    grp.add_argument('--light-reference-exposure', type=int, help='use the median valid rest sample of this exposure')

    a = ap.parse_args(argv)
    try:
        if a.cmd == 'list':
            print(f'store: {store_root()}')
            for ref in list_assets():
                asset = load_asset(ref)
                state = 'present' if asset.verify(hashes=False)['ok'] else 'not imported'
                print(f'{ref}  [{asset.manifest["status"]}]  tag={asset.hardware_tag}  {state}')
        elif a.cmd == 'show':
            print(json.dumps(load_asset(a.ref).manifest, indent=1))
        elif a.cmd == 'where':
            asset = load_asset(a.ref)
            print(json.dumps({'root': str(asset.root), **asset.replay_paths()}, indent=1))
        elif a.cmd == 'verify':
            report = load_asset(a.ref).verify(hashes=not a.quick)
            print(json.dumps(report, indent=1))
            return 0 if report['ok'] else 1
        elif a.cmd == 'import':
            report = load_asset(a.ref).import_from(a.source, overwrite=a.overwrite)
            print(json.dumps(report, indent=1))
            return 0 if report['ok'] else 1
        elif a.cmd == 'register':
            manifest = build_manifest(
                asset=a.asset, version=a.version, label=a.label, status=a.status, source_root=a.source,
                bundle=a.bundle, calibration_run=a.calibration_run, session_id=a.session_id, head=a.head,
                source_recording=a.source_recording,
                hardware_tag=a.hardware_tag,
                tag_assignment={'assigned_by': a.tag_assigned_by, 'note': a.tag_note} if a.hardware_tag else None,
                provenance={'source': 'TrilobiteVision', 'tv_revision': a.tv_revision,
                            'tv_config': a.tv_config, 'bundle': a.bundle,
                            'calibration_run': a.calibration_run,
                            'source_recording': a.source_recording, 'note': a.note})
            print(write_manifest(manifest, replace=a.replace))
            print(f"hardware_tag: {manifest['hardware_tag']}  ({manifest['hardware_tag_assignment'].get('source')})")
            for w in manifest['registration_warnings']:
                print(f'warning: {w}', file=sys.stderr)
        elif a.cmd == 'select':
            from .selection import SelectionError, select_calibration, summary_lines
            try:
                record = select_calibration(a.recording, a.head, explicit=a.calibration, default=a.default)
            except SelectionError as e:
                print(f'no calibration selected: {e}', file=sys.stderr)
                return 3
            for line in summary_lines(record):
                print(line)
            if a.out:
                a.out.parent.mkdir(parents=True, exist_ok=True)
                a.out.write_text(json.dumps(record, indent=1) + '\n')
                print(f'selection record: {a.out}')
        elif a.cmd == 'replay-light':
            import yaml
            from .recorded import RunOptions, run_recorded
            config = yaml.safe_load(a.config.read_text(encoding='utf-8'))
            selection = json.loads(a.selection.read_text(encoding='utf-8'))
            run = run_recorded(config, selection, a.out, RunOptions(
                bundle=a.bundle, fwhm_deg=a.fwhm_deg, enable_microsaccades=a.microsaccades,
                light_reference_counts=a.light_reference_counts,
                light_reference_exposure=a.light_reference_exposure))
            print(json.dumps({k: run[k] for k in ('calibration_selection', 'light_map', 'totals', 'fit_check', 'seconds')},
                             indent=1, default=str)[:4000])
            print(f'written: {a.out}')
    except AssetError as e:
        print(f'error: {e}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
