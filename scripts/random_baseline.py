"""Analytical chance, fixed-seed random choices, and constant-position baselines."""
import argparse
import collections
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from statistics_eval import cluster_interval, fixed_scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=20261012)
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding='utf-8'))
    questions = manifest['questions']
    records = [{'question_id': q['question_id'], 'raw_response': random.Random(
        f'{args.seed}/{q["question_id"]}').choice('ABCDE')} for q in questions]
    values, score = fixed_scores(questions, records)
    labels = collections.Counter(q['label'] for q in questions)
    result = {'condition': 'uniform_random_choice_without_model', 'seed': args.seed,
        'generator': 'Python random.Random seeded by seed/question_id',
        'expected_accuracy': .2, 'realized_score': score,
        'realized_score_video_bootstrap_95': cluster_interval(values, [q['video_id'] for q in questions]),
        'constant_option_accuracy': {letter: labels[letter] / len(questions) for letter in 'ABCDE'},
        'answer_position_counts': dict(labels), 'predictions': records,
        'interpretation': 'This is a random-selector baseline, not a VLM result. '
            'Theoretical expectation is exactly 20%; one realized draw can differ.'}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'predictions'}, indent=2))


if __name__ == '__main__':
    main()
