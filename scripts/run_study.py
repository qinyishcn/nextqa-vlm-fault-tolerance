"""Run the fixed LLaVA/NExT-QA study; require checkpoint and parity acceptance."""
import argparse
import collections
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
from safetensors.torch import load_file, save_file

from llava_runtime import LlavaRuntime, tensor_hash
from nextqa_eval import parse_answer
from prefix_cache import last_position_logits
from study_controls import condition_input


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def save_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes((json.dumps(value, indent=2) + '\n').encode())


def append_json(path, value):
    with Path(path).open('a', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def read_jsonl(path):
    if not Path(path).exists():
        return []
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line]


def encode_cache(runtime, assets, audit, output):
    folder = assets / 'pooled_features'
    folder.mkdir(parents=True, exist_ok=True)
    records = []
    identity = {'model_revision': '013210b3aff822f1558b166d39c1046dd109520f',
        'source_revision': 'bce12e479bc4dfee2b9c50c88137b01ff51bd483',
        'protocol': 'official-siglip-project-bilinear-bf16-batch4-v1'}
    runtime.vision_device(runtime.device)
    for metadata in audit['video_metadata']:
        video = metadata['video_id']
        target, saved = folder / f'{video}.safetensors', folder / f'{video}.json'
        if saved.exists():
            record = json.loads(saved.read_text())
            if (record['identity'] != identity or record['video_sha256'] != metadata['video_sha256']
                    or file_hash(target) != record['file_sha256']):
                raise ValueError(f'Existing feature cache identity mismatch: {video}')
        else:
            rows = sorted(metadata['frames'], key=lambda row: row['frame_index'])
            paths = [assets / 'frames' / video / row['file'] for row in rows]
            for path, row in zip(paths, rows):
                if file_hash(path) != row['png_sha256']:
                    raise ValueError(f'Native frame bytes changed before encoding: {video}')
            features = []
            start = time.monotonic()
            for begin in range(0, len(paths), 4):
                features.append(runtime.encode(paths[begin:begin + 4]))
            tensor = torch.cat(features, dim=0).contiguous()
            save_file({'features': tensor}, str(target))
            record = {'video_id': video, 'identity': identity,
                'video_sha256': metadata['video_sha256'], 'indices': [row['frame_index'] for row in rows],
                'shape': list(tensor.shape), 'dtype': 'bfloat16', 'file_sha256': file_hash(target),
                'feature_f32_sha256': tensor_hash(tensor), 'encode_seconds': time.monotonic() - start,
                'per_frame_f32_sha256': [tensor_hash(frame) for frame in tensor]}
            save_json(saved, record)
            del tensor, features
        records.append(record)
        print(f'Feature caches verified {len(records)}/60; video={video}', flush=True)
    save_json(output / 'feature_cache_audit.json', {'status': 'passed', 'videos': len(records), 'records': records})
    runtime.vision_device('cpu')


def evaluate(runtime, assets, manifest, plan, audit, output):
    # Full-model acceptance must exist before any formal answer attempt.
    acceptance = json.loads((output / 'pilot.json').read_text())
    if (acceptance['status'] != 'passed' or acceptance['plan_sha256'] != canonical_hash(plan)
            or acceptance['runtime'] != runtime.provenance()):
        raise ValueError('Formal inference requires matching pilot acceptance')
    identity = {'plan_sha256': canonical_hash(plan), 'manifest_sha256': canonical_hash(manifest),
        'runtime': runtime.provenance(), 'scope': 'offline feature erasure; no RTC/network emulation'}
    identity_path = output / 'run_identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError('Resume cannot mix runtime, plan or manifest identities')
    save_json(identity_path, identity)
    predictions = output / 'predictions'
    predictions.mkdir(parents=True, exist_ok=True)
    metadata = {row['video_id']: row for row in audit['video_metadata']}
    grouped = collections.defaultdict(list)
    for question in manifest['questions']:
        grouped[question['video_id']].append(question)
    expected = {q['question_id'] for q in manifest['questions']}
    completed = {}
    for condition in plan['conditions']:
        rows = read_jsonl(predictions / f'{condition["id"]}.jsonl')
        saved = {row['question_id']: row for row in rows}
        if len(saved) != len(rows) or not set(saved).issubset(expected):
            raise ValueError('Unexpected or duplicate formal prediction identity')
        completed[condition['id']] = saved
    attempted = {(row['condition_id'], row['question_id']) for row in read_jsonl(output / 'attempts.jsonl')}
    # An interrupted attempt remains wrong; a restart never replaces failed answers.
    for condition_id, question_id in sorted(attempted):
        if question_id not in expected or condition_id not in completed:
            raise ValueError('Unknown attempt identity')
        if question_id not in completed[condition_id]:
            row = {'condition_id': condition_id, 'question_id': question_id,
                'status': 'interrupted_attempt', 'raw_response': None, 'generated_token_ids': []}
            append_json(predictions / f'{condition_id}.jsonl', row)
            completed[condition_id][question_id] = row

    def load_cache(video):
        folder = assets / 'pooled_features'
        record = json.loads((folder / f'{video}.json').read_text())
        return {'features': load_file(str(folder / f'{video}.safetensors'))['features'], 'indices': record['indices']}

    runtime.vision_device('cpu')
    with last_position_logits(runtime.model):
        for video in sorted(grouped):
            cache = load_cache(video)
            questions = grouped[video]
            for condition in plan['conditions']:
                condition_id = condition['id']
                remaining = [q for q in questions if q['question_id'] not in completed[condition_id]]
                if not remaining:
                    continue
                pooled, prefix, suffixes = None, None, None
                try:
                    pooled, timing, control = condition_input(condition, video, metadata[video], cache,
                        metadata, load_cache, manifest['wrong_video_mapping'], plan['seed'])
                    torch.cuda.synchronize()
                    started = time.monotonic()
                    prefix, suffixes, receiver = runtime.build_prefix(questions, timing, pooled)
                    torch.cuda.synchronize()
                    prefix_id = f'{condition_id}/{video}'
                    append_json(output / 'prefix_audits.jsonl', {'prefix_id': prefix_id,
                        'condition_id': condition_id, 'video_id': video, 'control': control,
                        'receiver': receiver, 'prefill_seconds': time.monotonic() - started})
                except (RuntimeError, ValueError) as error:
                    for question in remaining:
                        row = {'condition_id': condition_id, 'question_id': question['question_id'],
                            'status': 'prefix_error', 'error_type': type(error).__name__,
                            'raw_response': None, 'generated_token_ids': []}
                        append_json(predictions / f'{condition_id}.jsonl', row)
                        completed[condition_id][question['question_id']] = row
                    del pooled, prefix, suffixes
                    gc.collect()
                    torch.cuda.empty_cache()
                    continue
                for index, question in enumerate(questions):
                    qid = question['question_id']
                    if qid in completed[condition_id]:
                        continue
                    append_json(output / 'attempts.jsonl', {'condition_id': condition_id,
                        'question_id': qid, 'prefix_id': prefix_id})
                    torch.cuda.synchronize()
                    started = time.monotonic()
                    try:
                        raw, tokens = runtime.answer_cached(suffixes[index], prefix)
                        torch.cuda.synchronize()
                        row = {'condition_id': condition_id, 'question_id': qid, 'prefix_id': prefix_id,
                            'status': 'answered', 'raw_response': raw, 'generated_token_ids': tokens,
                            'parsed_answer': parse_answer(raw), 'suffix_decode_seconds': time.monotonic() - started}
                    except RuntimeError as error:
                        row = {'condition_id': condition_id, 'question_id': qid, 'prefix_id': prefix_id,
                            'status': 'generation_error', 'error_type': type(error).__name__,
                            'raw_response': None, 'generated_token_ids': []}
                    append_json(predictions / f'{condition_id}.jsonl', row)
                    completed[condition_id][qid] = row
                del prefix, suffixes, pooled
                gc.collect()
                done = sum(len(rows) for rows in completed.values())
                print(f'Formal slots saved {done}/{len(plan["conditions"]) * len(expected)}; {prefix_id}', flush=True)
            del cache
    counts = {key: len(value) for key, value in completed.items()}
    save_json(output / 'completion.json', {'status': 'complete' if all(n == len(expected) for n in counts.values())
        else 'incomplete', 'fixed_slots': len(plan['conditions']) * len(expected),
        'saved_slots': sum(counts.values()), 'condition_counts': counts})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['source', 'model', 'assets', 'output', 'plan', 'manifest', 'frame-audit']:
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--stage', choices=['encode', 'evaluate', 'all'], default='all')
    args = parser.parse_args()
    sys.path.insert(0, args.source)
    assets, output = Path(args.assets), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    plan, manifest, audit = [json.loads(Path(path).read_text()) for path in [args.plan, args.manifest, args.frame_audit]]
    if canonical_hash(manifest) != plan['manifest_sha256']:
        raise ValueError('Manifest differs from the frozen pre-inference plan')
    runtime = LlavaRuntime(args.model)
    save_json(output / 'runtime_provenance.json', runtime.provenance())
    if args.stage in ['encode', 'all']:
        encode_cache(runtime, assets, audit, output)
    if args.stage in ['evaluate', 'all']:
        evaluate(runtime, assets, manifest, plan, audit, output)


if __name__ == '__main__':
    main()
