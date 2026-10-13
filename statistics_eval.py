"""Fixed-denominator scoring and video-cluster percentile intervals."""
import collections

import numpy as np

from nextqa_eval import parse_answer


def fixed_scores(questions, records):
    expected = {q['question_id']: q for q in questions}
    if len(expected) != len(questions):
        raise ValueError('Duplicate manifest question')
    saved = {}
    for record in records:
        qid = record['question_id']
        if qid not in expected or qid in saved:
            raise ValueError('Unexpected or duplicate result identity')
        saved[qid] = record
    values, missing, invalid, unavailable, errors = [], 0, 0, 0, 0
    for question in questions:
        record = saved.get(question['question_id'])
        if record is None:
            missing += 1
            unavailable += 1
            answer = None
        else:
            raw = record.get('raw_response')
            answer = parse_answer(raw)
            unavailable += raw is None
            invalid += raw is not None and answer is None
            errors += record.get('status', 'answered') != 'answered'
        values.append(int(answer == question['label']))
    return values, {'questions': len(questions), 'records_saved': len(saved),
        'answers_saved': sum(row.get('raw_response') is not None for row in saved.values()),
        'missing_answers': missing, 'unavailable_answers': unavailable, 'error_records': errors,
        'invalid_answers': invalid, 'correct': sum(values),
        'accuracy': sum(values) / len(questions)}


def cluster_interval(values, videos, confidence=.95, draws=10000, seed=20261012):
    if len(values) != len(videos) or not values or not 0 < confidence < 1:
        raise ValueError('Expected aligned nonempty values/videos and valid confidence')
    grouped = collections.defaultdict(list)
    for value, video in zip(values, videos):
        grouped[video].append(value)
    if len(grouped) < 2:
        return None
    groups = [grouped[key] for key in sorted(grouped)]
    totals = np.array([sum(group) for group in groups], dtype=np.float64)
    sizes = np.array([len(group) for group in groups])
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(groups), size=(draws, len(groups)))
    outcomes = totals[sampled].sum(axis=1) / sizes[sampled].sum(axis=1)
    alpha = (1 - confidence) / 2
    return np.quantile(outcomes, [alpha, 1 - alpha]).tolist()


def random_equivalence(interval90, chance=.2, margin=.05):
    if interval90 is None:
        return 'insufficient_video_clusters'
    low, high = interval90
    if low >= chance - margin and high <= chance + margin:
        return 'within_prespecified_5pp_equivalence_margin'
    if high < chance - margin or low > chance + margin:
        return 'outside_prespecified_5pp_equivalence_margin'
    return 'equivalence_not_established'


def summarize_results(questions, records, reference=None):
    """Score overall and type groups, retaining paired missing-answer penalties."""
    values, overall = fixed_scores(questions, records)
    reference_values = fixed_scores(questions, reference)[0] if reference is not None else None
    groups = {'overall': list(range(len(questions)))}
    for name in sorted({q['question_type'] for q in questions}):
        groups[name] = [i for i, q in enumerate(questions) if q['question_type'] == name]
    for name in sorted({q['question_type'][0] for q in questions}):
        groups[name] = [i for i, q in enumerate(questions) if q['question_type'][0] == name]
    result = {}
    for name, indices in groups.items():
        subset = [questions[i] for i in indices]
        identities = {q['question_id'] for q in subset}
        score = fixed_scores(subset, [row for row in records if row['question_id'] in identities])[1]
        outcomes = [values[i] for i in indices]
        videos = [q['video_id'] for q in subset]
        interval90 = cluster_interval(outcomes, videos, confidence=.90)
        score.update({'video_clusters': len(set(videos)),
            'video_bootstrap_95': cluster_interval(outcomes, videos),
            'video_bootstrap_90': interval90,
            'chance_accuracy': .2, 'chance_equivalence_margin': .05,
            'chance_equivalence': random_equivalence(interval90)})
        if reference_values is not None:
            differences = [values[i] - reference_values[i] for i in indices]
            score['paired_reference'] = {
                'reference_accuracy': sum(reference_values[i] for i in indices) / len(indices),
                'accuracy_difference': sum(differences) / len(differences),
                'difference_video_bootstrap_95': cluster_interval(differences, videos),
                'gained_correct': differences.count(1), 'lost_correct': differences.count(-1)}
        result[name] = score
    return {'overall': overall, 'groups': result,
        'interval_method': 'video-cluster percentile bootstrap; 10000 PCG64 draws; seed 20261012',
        'equivalence_method': '90% interval entirely within [15%,25%]; descriptive subgroup intervals unadjusted',
        'reference_score': fixed_scores(questions, reference)[1] if reference is not None else None}
