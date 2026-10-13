import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from statistics_eval import cluster_interval, fixed_scores, random_equivalence, summarize_results


class StatisticalEvidenceTests(unittest.TestCase):
    def test_missing_answer_keeps_original_denominator(self):
        questions = [{'question_id': 'q1', 'label': 'E'}, {'question_id': 'q2', 'label': 'A'}]
        values, score = fixed_scores(questions, [{'question_id': 'q1', 'raw_response': 'E'}])
        self.assertEqual(values, [1, 0])
        self.assertEqual((score['questions'], score['missing_answers'], score['accuracy']), (2, 1, .5))

    def test_unknown_result_is_not_accepted(self):
        with self.assertRaises(ValueError):
            fixed_scores([{'question_id': 'q1', 'label': 'A'}], [{'question_id': 'unknown'}])

    def test_saved_runtime_error_is_unavailable_rather_than_format_invalid(self):
        values, score = fixed_scores([{'question_id': 'q1', 'label': 'A'}],
            [{'question_id': 'q1', 'raw_response': None, 'status': 'generation_error'}])
        self.assertEqual(values, [0])
        self.assertEqual((score['records_saved'], score['answers_saved']), (1, 0))
        self.assertEqual((score['unavailable_answers'], score['invalid_answers'], score['error_records']), (1, 0, 1))

    def test_insufficient_video_clusters_do_not_imply_equivalence(self):
        self.assertIsNone(cluster_interval([1, 0], ['same', 'same']))
        self.assertEqual(random_equivalence(None), 'insufficient_video_clusters')

    def test_wide_interval_crossing_chance_is_not_equivalence(self):
        self.assertEqual(random_equivalence([.1, .3]), 'equivalence_not_established')
        self.assertEqual(random_equivalence([.18, .23]), 'within_prespecified_5pp_equivalence_margin')
        self.assertEqual(random_equivalence([.35, .5]), 'outside_prespecified_5pp_equivalence_margin')

    def test_paired_identical_predictions_have_zero_difference(self):
        self.assertEqual(cluster_interval([0] * 8, ['a'] * 3 + ['b'] * 5), [0, 0])

    def test_subgroups_and_paired_changes_keep_missing_questions(self):
        questions = [{'question_id': f'q{i}', 'label': label, 'video_id': video,
                      'question_type': kind} for i, label, video, kind in
                     [(1, 'A', 'v1', 'CW'), (2, 'B', 'v1', 'CH'),
                      (3, 'C', 'v2', 'TN'), (4, 'D', 'v2', 'TN')]]
        records = [{'question_id': f'q{i}', 'raw_response': answer}
                   for i, answer in [(1, 'A'), (2, 'E'), (3, 'C')]]
        reference = [{'question_id': f'q{i}', 'raw_response': answer}
                     for i, answer in [(2, 'B'), (3, 'C'), (4, 'D')]]
        summary = summarize_results(questions, records, reference)
        self.assertEqual(summary['overall']['missing_answers'], 1)
        self.assertEqual(summary['groups']['C']['questions'], 2)
        self.assertEqual(summary['groups']['T']['accuracy'], .5)
        paired = summary['groups']['overall']['paired_reference']
        self.assertEqual((paired['gained_correct'], paired['lost_correct']), (1, 2))
        self.assertEqual(paired['accuracy_difference'], -.25)


if __name__ == '__main__':
    unittest.main()
