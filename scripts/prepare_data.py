"""Pin official annotations and download only selected, hash-verified videos."""
import argparse
import concurrent.futures
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nextqa_eval import read_questions, select_video_groups, wrong_video_mapping

VAL_SHA256 = '43198bdef8436b8d64a9b75d846b0987c10cbf94ebf4be325c4a4e54634d66b8'


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def download_video(video, assets, catalogs):
    if not re.fullmatch(r'\d+', video):
        raise ValueError('Video ID must be numeric')
    chosen = None
    for repo, catalog, prefix in catalogs:
        name = f'{prefix}{video}.mp4'
        item = next((f for f in catalog['siblings'] if f['rfilename'] == name), None)
        if item:
            chosen = repo, catalog['sha'], name, item
            break
    if chosen is None:
        raise ValueError(f'No video source for selected ID {video}; do not drop this sample')
    repo, revision, name, item = chosen
    expected = item['lfs']['sha256']
    target = assets / 'videos' / f'{video}.mp4'
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists() or sha256(target) != expected:
        partial = target.with_suffix('.mp4.part')
        parts = assets / 'download_parts' / f'{video}.{expected[:12]}'
        parts.mkdir(parents=True, exist_ok=True)
        total = item['lfs']['size']
        session = requests.Session()
        session.trust_env = False
        with partial.open('wb') as assembled:
            for start in range(0, total, 1024 * 1024):
                end = min(start + 1024 * 1024 - 1, total - 1)
                part = parts / f'{start:012d}.bin'
                if not part.exists() or part.stat().st_size != end - start + 1:
                    for attempt in range(3):
                        try:
                            url = f'https://huggingface.co/datasets/{repo}/resolve/{revision}/{name}?download=true&part={start}'
                            with session.get(url, headers={'Range': f'bytes={start}-{end}'},
                                             timeout=(15, 30)) as response:
                                response.raise_for_status()
                                if (response.status_code != 206 or response.headers.get('Content-Range')
                                        != f'bytes {start}-{end}/{total}' or len(response.content) != end - start + 1):
                                    raise ValueError('Range not honored or truncated')
                                part.write_bytes(response.content)
                            break
                        except (requests.RequestException, ValueError) as error:
                            if attempt == 2:
                                raise RuntimeError(f'Video {video} range failed: {type(error).__name__}') from None
                            print(f'Retry video {video}, byte {start}: {type(error).__name__}', flush=True)
                            time.sleep(1 + attempt)
                assembled.write(part.read_bytes())
        if partial.stat().st_size != total or sha256(partial) != expected:
            raise ValueError(f'Video {video} size or SHA-256 mismatch')
        partial.replace(target)
    return {'video_id': video, 'repository': repo, 'revision': revision, 'path': name,
            'sha256': expected, 'bytes': target.stat().st_size,
            'provenance_note': 'Third-party video mirror; annotations from official NExT-QA. '
                'No claim of byte identity to the original Google Drive archive.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--target', type=int, default=500)
    parser.add_argument('--seed', type=int, default=20261012)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error('--workers must be between 1 and 16')
    assets, output = Path(args.assets), Path(args.output)
    annotation = assets / 'annotations/val.csv'
    if sha256(annotation) != VAL_SHA256:
        raise ValueError('Official pinned annotation bytes changed')
    all_questions = read_questions(annotation)
    questions, selection = select_video_groups(all_questions, args.target, args.seed)
    videos = sorted({q['video_id'] for q in questions})
    manifest = {'dataset': 'NExT-QA', 'split': 'validation',
        'official_repository': 'https://github.com/doc-doc/NExT-QA',
        'official_revision': '2432e9724f88ed9f40010e2989f104570a91de4e',
        'annotation_sha256': VAL_SHA256, 'selection': selection,
        'question_count': len(questions), 'video_count': len(videos), 'questions': questions,
        'wrong_video_mapping': wrong_video_mapping(videos, args.seed),
        'selection_before_inference': True, 'uniform_random_expected_accuracy': .2}
    output.mkdir(parents=True, exist_ok=True)
    (output / 'dataset_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({key: manifest[key] for key in ['question_count', 'video_count', 'selection']}, indent=2), flush=True)
    if args.download:
        catalogs = [('VLM2Vec/nextqa-rawvideo', json.loads((assets / 'video_catalog.json').read_text()), ''),
                    ('shuzhig/nextgqa', json.loads((assets / 'video_fallback_catalog.json').read_text()), 'videos/')]
        records = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(download_video, video, assets, catalogs) for video in videos]
            for future in concurrent.futures.as_completed(futures):
                records.append(future.result())
                print(f'Hash-verified videos {len(records)}/{len(videos)}', flush=True)
        (output / 'video_provenance.json').write_text(
            json.dumps(sorted(records, key=lambda r: r['video_id']), indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
