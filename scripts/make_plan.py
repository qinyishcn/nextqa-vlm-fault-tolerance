"""Freeze the complete condition matrix before any model accuracy is scored."""
import argparse
import hashlib
import json
from pathlib import Path


def build_plan(manifest):
    conditions = [
        {'id': 'text_only', 'kind': 'text', 'frames': 0},
        {'id': 'text_timing_16', 'kind': 'text_timing', 'frames': 16, 'scheme': 'uniform'},
    ]
    for scheme in ['native_center', 'uniform']:
        for frames in [1, 4, 16, 64]:
            if scheme == 'uniform' and frames == 1:
                continue  # Identical middle-frame input, already native_center_1.
            conditions.append({'id': f'{scheme}_{frames}', 'kind': 'clean',
                               'frames': frames, 'scheme': scheme})
    for kind in ['repeat', 'shuffle', 'reverse', 'wrong_video', 'gaussian', 'zero']:
        conditions.append({'id': f'{kind}_16', 'kind': kind, 'frames': 16, 'scheme': 'uniform'})
    rates = [0, .05, .10, .20, .30, .40, .50, .60, .70, .80, .90, .95, 1]
    for rate in rates[1:-1]:
        for seed in range(3):
            conditions.append({'id': f'loss_{round(rate * 100):03d}_s{seed}', 'kind': 'loss',
                'frames': 16, 'scheme': 'uniform', 'loss_rate': rate, 'mask_repeat': seed})
    return {'model': 'lmms-lab/LLaVA-Video-7B-Qwen2',
        'model_revision': '013210b3aff822f1558b166d39c1046dd109520f',
        'source_revision': 'bce12e479bc4dfee2b9c50c88137b01ff51bd483',
        'manifest_sha256': hashlib.sha256(json.dumps(json.loads(manifest), sort_keys=True,
            separators=(',', ':')).encode()).hexdigest(),
        'manifest_identity_encoding': 'UTF-8 json.dumps(sort_keys=True,separators=(comma,colon)); ensure_ascii=True',
        'seed': 20261012, 'wire_feature_packet_bytes': 1080,
        'feature_dtype': 'little-endian F32', 'feature_boundary': 'get_2dPool output, before grid newline',
        'reference_frames_for_tolerance': 16, 'loss_rates': rates, 'loss_mask_repeats': 3,
        'endpoint_aliases': {'0': 'uniform_16', '1': 'zero_16'},
        'conditions': conditions, 'questions_per_condition': 495,
        'missing_invalid_answers': 'count wrong, fixed manifest denominator; no replacement reruns',
        'packet_model': 'offline independent packet erasure; shared mask per video/repeat; nested masks across rates',
        'gaussian_control': 'seeded independent N(0,1), BF16 rounding; out-of-distribution control',
        'timing_control': 'same chronological slot timestamps for matched 16-frame ablations',
        'generation': {'do_sample': False, 'num_beams': 1, 'repetition_penalty': 1.0, 'max_new_tokens': 8},
        'runtime_optimizations': {'encoder_batch_frames': 4, 'prefill_chunk_tokens': 1024,
            'shared_video_prefix_cache': True, 'last_position_lm_head': True},
        'training': False, 'quantization': 'NF4 double quantization, BF16 compute; LM linear layers only',
        'score_scope': 'fixed selected 60-video / 495-question subset; not all-validation estimate'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    raw = Path(args.manifest).read_bytes()
    plan = build_plan(raw)
    manifest = json.loads(raw)
    if manifest['question_count'] != 495 or manifest['video_count'] != 60:
        raise ValueError('This preregistration is for the confirmed fixed 495-question subset')
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # LF bytes make the file identity independent of Windows text newline conversion.
    output.write_bytes((json.dumps(plan, indent=2) + '\n').encode())
    print(json.dumps({'conditions': len(plan['conditions']),
        'fixed_question_slots': len(plan['conditions']) * manifest['question_count'],
        'plan_sha256': hashlib.sha256(output.read_bytes()).hexdigest()}, indent=2))


if __name__ == '__main__':
    main()
