"""
python -m rhabdoforge.hardware list
python -m rhabdoforge.hardware show   ASSET@VERSION
python -m rhabdoforge.hardware where  ASSET@VERSION
python -m rhabdoforge.hardware verify ASSET@VERSION [--quick]
python -m rhabdoforge.hardware import ASSET@VERSION --from <TrilobiteVision root>
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
                provenance={'source': 'TrilobiteVision', 'tv_revision': a.tv_revision,
                            'tv_config': a.tv_config, 'bundle': a.bundle,
                            'calibration_run': a.calibration_run,
                            'source_recording': a.source_recording, 'note': a.note})
            print(write_manifest(manifest))
    except AssetError as e:
        print(f'error: {e}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
