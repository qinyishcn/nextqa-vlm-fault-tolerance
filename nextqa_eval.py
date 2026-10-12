"""Model-independent data, sampling, scoring and feature-erasure primitives."""
import collections
import csv
import hashlib
import math
import random
import re

QUESTION_TYPES = {'CW', 'CH', 'TN', 'TC', 'TP', 'DL', 'DC', 'DO'}


def read_questions(path):
    questions, seen = [], set()
    with open(path, encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            qid = f'{row["video"]}_{row["qid"]}'
            index = int(row['answer'])
            choices = [row[f'a{i}'] for i in range(5)]
            if (qid in seen or index not in range(5) or row['type'] not in QUESTION_TYPES
                    or not row['question'] or any(not value for value in choices)):
                raise ValueError(f'Invalid or duplicate benchmark record: {qid}')
            seen.add(qid)
            questions.append({'question_id': qid, 'video_id': row['video'],
                'question': row['question'], 'choices': choices, 'label': 'ABCDE'[index],
                'question_type': row['type'], 'annotation_frame_count': int(row['frame_count']),
                'annotation_width': int(row['width']), 'annotation_height': int(row['height'])})
    return questions


def select_video_groups(questions, target, seed):
    """Stratified cluster sample; every selected video's questions stay together.

    Assign each video to its globally rarest question type. For each possible
    number of videos allocate at least one per stratum, then proportional quotas
    by largest remainders. Pick the size nearest the target question count.
    No labels or model outputs are consulted.
    """
    if not questions or target <= 0:
        raise ValueError('Expected nonempty questions and positive target')
    by_video = collections.defaultdict(list)
    for question in questions:
        by_video[question['video_id']].append(question)
    if target >= len(questions):
        return list(questions), {'method': 'full validation', 'seed': seed,
                                 'target_questions': target, 'videos': len(by_video)}
    frequencies = collections.Counter(q['question_type'] for q in questions)
    strata = collections.defaultdict(list)
    for video, rows in sorted(by_video.items()):
        rarest = min({q['question_type'] for q in rows}, key=lambda t: (frequencies[t], t))
        strata[rarest].append(video)
    rng = random.Random(seed)
    for name in sorted(strata):
        rng.shuffle(strata[name])
    total_videos, best = len(by_video), None
    for number in range(len(strata), total_videos + 1):
        remaining = number - len(strata)
        capacities = {name: len(videos) - 1 for name, videos in strata.items()}
        total_capacity = total_videos - len(strata)
        ideal = {name: remaining * capacity / total_capacity if total_capacity else 0
                 for name, capacity in capacities.items()}
        quotas = {name: 1 + math.floor(value) for name, value in ideal.items()}
        extras = number - sum(quotas.values())
        order = sorted(strata, key=lambda name: (-(ideal[name] % 1), name))
        for name in order[:extras]:
            quotas[name] += 1
        chosen = {video for name, videos in strata.items() for video in videos[:quotas[name]]}
        count = sum(len(by_video[video]) for video in chosen)
        key = (abs(count - target), count > target, number)
        if best is None or key < best[0]:
            best = key, chosen, quotas
    _, chosen, quotas = best
    selected = [q for q in questions if q['video_id'] in chosen]
    return selected, {'method': 'rarest-type stratified video-cluster sample', 'seed': seed,
        'target_questions': target, 'videos': len(chosen), 'video_stratum_quotas': quotas,
        'selected_type_counts': dict(collections.Counter(q['question_type'] for q in selected)),
        'source_type_counts': dict(frequencies)}


def question_prompt(question):
    # Whitelist fields so labels and dataset identifiers cannot enter the prompt.
    options = '\n'.join(f'{letter}. {answer}' for letter, answer in zip('ABCDE', question['choices']))
    return ('Answer the multiple-choice question. Respond with exactly one option letter '
            '(A, B, C, D, or E).\nQuestion: ' + question['question'] + '\n' + options)


def parse_answer(raw):
    match = re.fullmatch(r'\s*\(?([ABCDE])\)?[.)]?\s*', raw) if isinstance(raw, str) else None
    return match.group(1) if match else None


def frame_indices(total_frames, requested, scheme):
    if total_frames < 1 or requested < 1:
        raise ValueError('Frame counts must be positive')
    count = min(total_frames, requested)
    if scheme == 'native_center':
        start = (total_frames - count) // 2
        return list(range(start, start + count))
    if scheme == 'uniform':
        if count == 1:
            return [(total_frames - 1) // 2]
        return [round(i * (total_frames - 1) / (count - 1)) for i in range(count)]
    raise ValueError(f'Unknown frame selection: {scheme}')


def wrong_video_mapping(video_ids, seed):
    videos = sorted(set(video_ids))
    if len(videos) < 2:
        raise ValueError('Wrong-video control requires at least two videos')
    random.Random(seed).shuffle(videos)
    return {video: videos[(i + 1) % len(videos)] for i, video in enumerate(videos)}


def packet_erasure(payload, rate, seed, packet_feature_bytes=1080):
    """IID packet erasure on little-endian F32 bytes; common seed pairs loss rates."""
    if (not 0 <= rate <= 1 or packet_feature_bytes <= 0
            or packet_feature_bytes % 4 or len(payload) % 4):
        raise ValueError('Expected a rate in [0,1] and F32-aligned feature bytes')
    rng = random.Random(seed)
    damaged = bytearray(payload)
    count = math.ceil(len(payload) / packet_feature_bytes)
    mask = bytearray(math.ceil(count / 8))
    missing = zeroed = 0
    for index in range(count):
        if rng.random() < rate:
            mask[index // 8] |= 1 << (index % 8)
            begin, end = index * packet_feature_bytes, min((index + 1) * packet_feature_bytes, len(payload))
            damaged[begin:end] = bytes(end - begin)
            missing += 1
            zeroed += end - begin
    return bytes(damaged), {'algorithm': 'python-random-v1-bernoulli-lsb-bitset',
        'requested_rate': rate, 'seed': seed, 'packet_feature_bytes': packet_feature_bytes,
        'feature_bytes': len(payload), 'total_packets': count, 'missing_packets': missing,
        'zeroed_bytes': zeroed, 'mask_sha256': hashlib.sha256(mask).hexdigest(),
        'original_feature_sha256': hashlib.sha256(payload).hexdigest(),
        'received_feature_sha256': hashlib.sha256(damaged).hexdigest()}
