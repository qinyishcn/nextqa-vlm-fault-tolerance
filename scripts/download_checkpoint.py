"""Download an immutable HF checkpoint with resumable ranges and content hashes."""
import argparse
import concurrent.futures
import hashlib
import json
import threading
import time
from pathlib import Path

import requests

REPOSITORY = 'lmms-lab/LLaVA-Video-7B-Qwen2'
REVISION = '013210b3aff822f1558b166d39c1046dd109520f'
MS_REPOSITORY = 'QwenCollection/LLaVA-NeXT-Video-7B-Qwen2'
MS_REVISION = 'd21b696615745ad7779b1d8ef9d74bc48ef23ad4'
LOCAL = threading.local()
MIRROR_URLS = {}
MIRROR_LOCK = threading.Lock()


def session():
    if not hasattr(LOCAL, 'session'):
        LOCAL.session = requests.Session()
        LOCAL.session.trust_env = False
    return LOCAL.session


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def part_download(repository, revision, row, start, block_size, folder, mirror):
    total = row['lfs']['size']
    end = min(start + block_size, total) - 1
    path = folder / f'{start:012d}.bin'
    if path.exists() and path.stat().st_size == end - start + 1:
        return path.stat().st_size
    for attempt in range(3):
        try:
            url = (f'https://huggingface.co/{repository}/resolve/{revision}/'
                   f'{row["rfilename"]}?download=true&part={start}')
            if mirror == 'modelscope':
                with MIRROR_LOCK:
                    if row['rfilename'] not in MIRROR_URLS:
                        endpoint = (f'https://www.modelscope.cn/api/v1/models/{MS_REPOSITORY}/repo'
                            f'?Revision={MS_REVISION}&FilePath={row["rfilename"]}')
                        with session().get(endpoint, headers={'Range': 'bytes=0-0'},
                                stream=True, timeout=(15, 60)) as redirect:
                            redirect.raise_for_status()
                            MIRROR_URLS[row['rfilename']] = redirect.url
                    url = MIRROR_URLS[row['rfilename']]
            with session().get(url, headers={'Range': f'bytes={start}-{end}'},
                               timeout=(15, 120)) as response:
                response.raise_for_status()
                if (response.status_code != 206 or response.headers.get('Content-Range')
                        != f'bytes {start}-{end}/{total}' or len(response.content) != end - start + 1):
                    raise ValueError('Checkpoint range was not honored or was truncated')
                pending = path.with_suffix('.part')
                pending.write_bytes(response.content)
                pending.replace(path)
            return end - start + 1
        except (requests.RequestException, ValueError) as error:
            if attempt == 2:
                raise RuntimeError(f'{row["rfilename"]} range {start}: {type(error).__name__}') from None
            print(f'Retry {row["rfilename"]} range {start}: {type(error).__name__}', flush=True)
            time.sleep(attempt + 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--provenance', required=True)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--block-mib', type=int, default=4)
    parser.add_argument('--mirror', choices=['huggingface', 'modelscope'], default='huggingface')
    args = parser.parse_args()
    if not 1 <= args.workers <= 24 or not 1 <= args.block_mib <= 32:
        parser.error('Expected workers in 1..24 and block size in 1..32 MiB')
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    with session().get(f'https://huggingface.co/api/models/{REPOSITORY}/revision/{REVISION}?blobs=true',
                       timeout=(15, 60)) as response:
        response.raise_for_status()
        catalog = response.json()
    if catalog['sha'] != REVISION:
        raise ValueError('Checkpoint revision mismatch')
    if args.mirror == 'modelscope':
        with session().get(f'https://www.modelscope.cn/api/v1/models/{MS_REPOSITORY}/repo/files',
                params={'Revision': MS_REVISION, 'Recursive': 'true'}, timeout=(15, 60)) as response:
            response.raise_for_status()
            mirror_files = {row['Path']: row for row in response.json()['Data']['Files']}
        for row in catalog['siblings']:
            if row['rfilename'].endswith('.safetensors'):
                copied = mirror_files[row['rfilename']]
                if copied['Sha256'] != row['lfs']['sha256'] or copied['Size'] != row['lfs']['size']:
                    raise ValueError('Mirror weights differ from the pinned official checkpoint')
    excluded = {'.gitattributes', 'trainer_state.json', 'training_args.bin'}
    rows = [row for row in catalog['siblings'] if row['rfilename'] not in excluded]
    records, pending, jobs = [], [], []
    block_size = args.block_mib * 1024 * 1024
    for row in rows:
        name = row['rfilename']
        if Path(name).name != name:
            raise ValueError('Unexpected nested checkpoint filename')
        path = root / name
        if 'lfs' not in row:
            data = path.read_bytes() if path.exists() else session().get(
                f'https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{name}', timeout=(15, 60)).content
            git_hash = hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()
            if len(data) != row['size'] or git_hash != row['blobId']:
                raise ValueError(f'Checkpoint metadata identity mismatch: {name}')
            path.write_bytes(data)
            records.append({'path': name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
        elif path.exists() and digest(path) == row['lfs']['sha256']:
            records.append({'path': name, 'bytes': path.stat().st_size, 'sha256': row['lfs']['sha256']})
        else:
            folder = root / 'download_parts' / f'{name}.{row["lfs"]["sha256"][:12]}'
            folder.mkdir(parents=True, exist_ok=True)
            pending.append((row, folder))
            jobs.extend((row, start, folder) for start in range(0, row['lfs']['size'], block_size))
    completed, total = 0, sum(row['lfs']['size'] for row, _ in pending)
    print(f'Pinned checkpoint: {REPOSITORY}@{REVISION}; {total} bytes pending', flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(part_download, REPOSITORY, REVISION, row, start, block_size, folder, args.mirror)
                   for row, start, folder in jobs]
        last_print = time.monotonic()
        for future in concurrent.futures.as_completed(futures):
            completed += future.result()
            if time.monotonic() - last_print > 20 or completed == total:
                print(f'Checkpoint range bytes ready: {completed}/{total}', flush=True)
                last_print = time.monotonic()
    for row, folder in pending:
        target = root / row['rfilename']
        assembled = target.with_suffix('.safetensors.part')
        with assembled.open('wb') as stream:
            for start in range(0, row['lfs']['size'], block_size):
                stream.write((folder / f'{start:012d}.bin').read_bytes())
        if assembled.stat().st_size != row['lfs']['size'] or digest(assembled) != row['lfs']['sha256']:
            raise ValueError(f'Checkpoint whole-file hash mismatch: {row["rfilename"]}')
        assembled.replace(target)
        records.append({'path': row['rfilename'], 'bytes': target.stat().st_size,
                        'sha256': row['lfs']['sha256']})
        print(f'Checkpoint SHA-256 verified: {target.name}', flush=True)
    provenance = {'repository': REPOSITORY, 'revision': REVISION, 'files': sorted(records, key=lambda x: x['path'])}
    provenance['weight_download_mirror'] = ({'repository': MS_REPOSITORY, 'revision': MS_REVISION,
        'all_weight_sha256_equal_pinned_huggingface': True} if args.mirror == 'modelscope' else None)
    output = Path(args.provenance)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(provenance, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
