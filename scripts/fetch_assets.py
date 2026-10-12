"""Fetch pinned official annotations and third-party video catalogs (no weights)."""
import argparse
import hashlib
import json
from pathlib import Path

import requests

REVISION = '2432e9724f88ed9f40010e2989f104570a91de4e'
ANNOTATIONS = {
    'dataset/nextqa/val.csv': '43198bdef8436b8d64a9b75d846b0987c10cbf94ebf4be325c4a4e54634d66b8',
    'dataset/nextqa/map_vid_vidorID.json': '1955d89e7a175346faf8d4a582297ff349619231394f59fd7872f459ac8a8685',
    'LICENSE': '7ba077d25ffc3137200430d2a6b82b81d68671dfda732b5e8582f8e632ef8714',
    'README.md': '8f9853e632ba2aa39de02e5e8fd3718134192552c19ce2bb5af0a167f8ecb37b',
}
CATALOGS = [
    ('VLM2Vec/nextqa-rawvideo', 'b454fdb10e211887097207db8dd74fcfa7b486ab', 'video_catalog.json'),
    ('shuzhig/nextgqa', '3fc4ace54d7566f46145ec29138e912e557ae870', 'video_fallback_catalog.json'),
]


def fetch(session, url):
    with session.get(url, timeout=(15, 60)) as response:
        response.raise_for_status()
        return response.content


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', required=True)
    parser.add_argument('--output', required=True, help='Public provenance directory')
    args = parser.parse_args()
    assets, output = Path(args.assets), Path(args.output)
    folder = assets / 'annotations'
    folder.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.trust_env = False
    records = []
    for source, expected in ANNOTATIONS.items():
        target = folder / Path(source).name
        data = target.read_bytes() if target.exists() else fetch(session,
            f'https://raw.githubusercontent.com/doc-doc/NExT-QA/{REVISION}/{source}')
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f'Pinned annotation SHA-256 mismatch: {source}')
        target.write_bytes(data)
        records.append({'path': source, 'sha256': expected, 'bytes': len(data)})
    provenance = {'repository': 'doc-doc/NExT-QA', 'revision': REVISION, 'files': records}
    for destination in [assets, output]:
        (destination / 'annotation_provenance.json').write_text(
            json.dumps(provenance, indent=2), encoding='utf-8')
    for repository, revision, filename in CATALOGS:
        target = assets / filename
        catalog = json.loads(target.read_bytes() if target.exists() else fetch(session,
            f'https://huggingface.co/api/datasets/{repository}/revision/{revision}?blobs=true'))
        if catalog.get('sha') != revision or not isinstance(catalog.get('siblings'), list):
            raise ValueError(f'Pinned video catalog identity mismatch: {repository}')
        target.write_text(json.dumps(catalog), encoding='utf-8')
        print(f'Catalog verified: {repository}@{revision}', flush=True)
    print(f'Official annotations verified: {len(records)} files', flush=True)


if __name__ == '__main__':
    main()
