"""Independently replay received feature bytes and re-score a completed study."""
import argparse
import collections
import csv
import functools
import hashlib
import json
import random
import re
from pathlib import Path

import torch
from safetensors.torch import load_file


def digest(value):
    return hashlib.sha256(value).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def wire(tensor):
    return tensor.float().contiguous().numpy().astype('<f4', copy=False).tobytes()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['assets', 'results', 'manifest', 'plan']:
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    root, assets = Path(args.results), Path(args.assets)
    manifest = json.loads(Path(args.manifest).read_text())
    plan = json.loads(Path(args.plan).read_text())
    completion = json.loads((root / 'completion.json').read_text())
    assert completion['status'] == 'complete'
    questions = manifest['questions']
    expected = {q['question_id']: q for q in questions}
    annotation_path = assets / 'annotations' / 'val.csv'
    assert digest(annotation_path.read_bytes()) == '43198bdef8436b8d64a9b75d846b0987c10cbf94ebf4be325c4a4e54634d66b8'
    with annotation_path.open(encoding='utf-8-sig', newline='') as stream:
        official = {f"{r['video']}_{r['qid']}": r for r in csv.DictReader(stream)}
    for q in questions:
        original = official[q['question_id']]
        assert q['question'] == original['question']
        assert q['choices'] == [original[f'a{i}'] for i in range(5)]
        assert q['label'] == 'ABCDE'[int(original['answer'])]
        assert q['question_type'] == original['type']

    analysis = json.loads((root / 'analysis.json').read_text())
    scores, statuses, slots, answer_prefixes = {}, collections.Counter(), 0, set()
    for condition in plan['conditions']:
        rows = read_jsonl(root / 'predictions' / f"{condition['id']}.jsonl")
        assert len(rows) == len(expected) and {r['question_id'] for r in rows} == set(expected)
        correct = 0
        for row in rows:
            assert row['condition_id'] == condition['id']
            statuses[row['status']] += 1
            if row['status'] == 'answered':
                assert row['prefix_id'] == f"{condition['id']}/{expected[row['question_id']]['video_id']}"
                answer_prefixes.add(row['prefix_id'])
            raw = row.get('raw_response')
            match = re.fullmatch(r'\s*\(?([ABCDE])\)?[.)]?\s*', raw) if isinstance(raw, str) else None
            answer = match.group(1) if match else None
            correct += answer == expected[row['question_id']]['label']
        assert correct == analysis['condition_summaries'][condition['id']]['overall']['correct']
        scores[condition['id']] = {'correct': correct, 'questions': len(rows)}
        slots += len(rows)
    assert slots == completion['fixed_slots'] == completion['saved_slots']

    retained = 0
    checkpoint = json.loads((root / 'resume_checkpoint.json').read_text())
    for item in checkpoint['saved_prediction_files']:
        lines = (root / 'predictions' / item['path']).read_bytes().splitlines(keepends=True)
        assert digest(b''.join(lines[:item['rows']])) == item['sha256']
        retained += item['rows']
    assert retained == checkpoint['saved_slots_before_resume']

    @functools.lru_cache(maxsize=3)
    def cache(video):
        folder = assets / 'pooled_features'
        metadata = json.loads((folder / f'{video}.json').read_text())
        return load_file(str(folder / f'{video}.safetensors'))['features'], metadata['indices']

    def select(video, indices):
        tensor, source_indices = cache(video)
        positions = {index: position for position, index in enumerate(source_indices)}
        return tensor.index_select(0, torch.tensor([positions[index] for index in indices]))

    audits = {}
    for row in read_jsonl(root / 'prefix_audits.jsonl'):
        if row['prefix_id'] in audits:
            previous = audits[row['prefix_id']]
            assert previous['control'] == row['control']
            assert previous['receiver']['prefix_embedding_sha256'] == row['receiver']['prefix_embedding_sha256']
        audits[row['prefix_id']] = row
    assert answer_prefixes.issubset(audits)
    conditions = {c['id']: c for c in plan['conditions']}
    replay, masks = [], 0
    for row in audits.values():
        control, condition = row['control'], conditions[row['condition_id']]
        if condition['kind'] in ['text', 'text_timing']:
            continue
        video = row['video_id']
        target = select(video, control['target_frame_indices'])
        assert digest(wire(target)) == control['clean_target_feature_sha256']
        if condition['kind'] == 'wrong_video':
            assert control['source_video_id'] == manifest['wrong_video_mapping'][video] != video
        feature = select(control['source_video_id'], control['source_frame_indices'])
        if condition['kind'] == 'gaussian':
            seed = int.from_bytes(hashlib.sha256(f"{plan['seed']}/{video}/gaussian".encode()).digest()[:8], 'little')
            feature = torch.randn(feature.shape, generator=torch.Generator().manual_seed(seed),
                                  dtype=torch.float32).to(torch.bfloat16)
        payload = wire(feature)
        if condition['kind'] in ['zero', 'loss']:
            erasure = control['erasure']
            rate = 1 if condition['kind'] == 'zero' else condition['loss_rate']
            seed = f"{plan['seed']}/{video}/mask_repeat_{condition.get('mask_repeat', 0)}"
            assert erasure['seed'] == seed and erasure['requested_rate'] == rate
            packet_bytes = plan['wire_feature_packet_bytes']
            assert packet_bytes == erasure['packet_feature_bytes']
            assert digest(payload) == erasure['original_feature_sha256']
            count = (len(payload) + packet_bytes - 1) // packet_bytes
            rng, mask, damaged = random.Random(seed), bytearray((count + 7) // 8), bytearray(payload)
            missing = zeroed = 0
            for index in range(count):
                if rng.random() < rate:
                    mask[index // 8] |= 1 << (index % 8)
                    begin, end = index * packet_bytes, min((index + 1) * packet_bytes, len(payload))
                    damaged[begin:end] = b'\0' * (end - begin)
                    missing, zeroed = missing + 1, zeroed + end - begin
            assert digest(mask) == erasure['mask_sha256']
            assert count == erasure['total_packets'] and missing == erasure['missing_packets']
            assert zeroed == erasure['zeroed_bytes'] and len(payload) == erasure['feature_bytes']
            payload = damaged
            masks += 1
        actual = digest(payload)
        assert actual == control['condition_feature_sha256']
        assert actual == row['receiver']['receiver']['received_sha256']
        replay.append({'prefix_id': row['prefix_id'], 'received_feature_sha256': actual, 'passed': True})
        if len(replay) % 100 == 0:
            print(f'Independent feature replay passed {len(replay)}', flush=True)
    expected_visual = len({q['video_id'] for q in questions}) * sum(
        c['kind'] not in ['text', 'text_timing'] for c in plan['conditions'])
    assert len(replay) == expected_visual
    result = {'status': 'passed', 'fixed_slots_independently_scored': slots,
        'official_annotation_questions_matched': len(questions), 'saved_answers_retained_byte_for_byte': retained,
        'status_counts': dict(statuses), 'conditions': scores, 'received_feature_buffers_replayed': len(replay),
        'packet_masks_replayed': masks, 'replay': replay,
        'scope': 'CPU replay of cached source features and declared damage; no new model answers generated'}
    (root / 'independent_verification.json').write_bytes((json.dumps(result, indent=2) + '\n').encode())
    print(json.dumps({key: value for key, value in result.items() if key not in ['replay', 'conditions']}, indent=2))


if __name__ == '__main__':
    main()
