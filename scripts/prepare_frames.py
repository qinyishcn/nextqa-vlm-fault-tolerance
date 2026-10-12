"""Cache exact decoded native frames, without model-specific resizing/sampling."""
import argparse
import concurrent.futures
import hashlib
import json
import sys
from pathlib import Path

import av

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nextqa_eval import frame_indices


def file_sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def extract(video, assets, provenance, annotation):
    source = assets / 'videos' / f'{video}.mp4'
    digest = file_sha(source)
    if digest != provenance['sha256']:
        raise ValueError(f'Video hash changed: {video}')
    folder = assets / 'frames' / video
    folder.mkdir(parents=True, exist_ok=True)
    metadata_path = folder / 'frames.json'
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata['video_sha256'] == digest and all(
                file_sha(folder / f['file']) == f['png_sha256'] for f in metadata['frames']):
            return metadata
        raise ValueError(f'Existing frame-cache identity mismatch: {video}')
    with av.open(str(source)) as container:
        stream = container.streams.video[0]
        stream.thread_count = 1
        total, fps = stream.frames, float(stream.average_rate)
        width, height = stream.width, stream.height
    if not total:
        with av.open(str(source)) as container:
            total = sum(1 for _ in container.decode(video=0))
    selections = {f'{scheme}_{number}': frame_indices(total, number, scheme)
                  for scheme in ['native_center', 'uniform'] for number in [1, 4, 16, 64]}
    wanted = set(index for indices in selections.values() for index in indices)
    frames, decoded = [], 0
    with av.open(str(source)) as container:
        container.streams.video[0].thread_count = 1
        for index, frame in enumerate(container.decode(video=0)):
            decoded += 1
            if index not in wanted:
                continue
            image = frame.to_image()
            path = folder / f'{index:06d}.png'
            image.save(path, compress_level=2)
            frames.append({'frame_index': index, 'time_seconds': float(frame.time),
                'file': path.name, 'width': image.width, 'height': image.height,
                'rgb_sha256': hashlib.sha256(image.tobytes()).hexdigest(),
                'png_sha256': file_sha(path)})
    if decoded != total or len(frames) != len(wanted):
        raise ValueError(f'Frame count mismatch before inference: {video}')
    metadata = {'video_id': video, 'video_sha256': digest, 'decoded_frame_count': decoded,
        'average_fps': fps, 'width': width, 'height': height, 'decoder': f'PyAV {av.__version__}',
        'cache_format': 'lossless RGB PNG; no model resizing',
        'annotation_frame_count': annotation['annotation_frame_count'],
        'annotation_width': annotation['annotation_width'], 'annotation_height': annotation['annotation_height'],
        'annotation_metadata_matches': decoded == annotation['annotation_frame_count']
            and width == annotation['annotation_width'] and height == annotation['annotation_height'],
        'selections': selections, 'frames': frames}
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--provenance', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    assets = Path(args.assets)
    manifest = json.loads(Path(args.manifest).read_text(encoding='utf-8'))
    sources = {row['video_id']: row for row in json.loads(Path(args.provenance).read_text())}
    annotations = {row['video_id']: row for row in manifest['questions']}
    result = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(extract, video, assets, sources[video], row)
                   for video, row in sorted(annotations.items())]
        for future in concurrent.futures.as_completed(futures):
            result.append(future.result())
            print(f'Native-frame caches verified {len(result)}/{len(annotations)}', flush=True)
    for metadata in result:
        times = {frame['frame_index']: frame['time_seconds'] for frame in metadata['frames']}
        metadata['selection_time_spans_seconds'] = {
            name: times[indices[-1]] - times[indices[0]]
            for name, indices in metadata['selections'].items()}
    summary = {'status': 'passed', 'videos': len(result),
        'cached_native_frames': sum(len(row['frames']) for row in result),
        'annotation_metadata_mismatches': [row['video_id'] for row in result if not row['annotation_metadata_matches']],
        'video_metadata': sorted(result, key=lambda row: row['video_id'])}
    Path(args.output).write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in summary.items() if k != 'video_metadata'}, indent=2))


if __name__ == '__main__':
    main()
