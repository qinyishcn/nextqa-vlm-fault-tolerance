import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nextqa_eval import (frame_indices, packet_erasure, parse_answer, question_prompt,
                         select_video_groups, wrong_video_mapping)


class EvaluationControlsTests(unittest.TestCase):
    def test_labels_and_identifiers_do_not_enter_prompt(self):
        q = {'question': 'What happened?', 'choices': ['one', 'two', 'three', 'four', 'five'],
             'label': 'SECRET_LABEL', 'video_id': 'SECRET_VIDEO', 'question_id': 'SECRET_ID'}
        prompt = question_prompt(q)
        self.assertNotIn('SECRET', prompt)
        self.assertIn('E. five', prompt)

    def test_invalid_or_missing_outputs_are_not_guessed(self):
        for raw in [None, '', 'B or C', 'A. an explanation', 'F']:
            self.assertIsNone(parse_answer(raw))
        self.assertEqual(parse_answer(' (E) '), 'E')

    def test_consecutive_and_whole_clip_indices_are_distinct(self):
        self.assertEqual(frame_indices(100, 4, 'native_center'), [48, 49, 50, 51])
        self.assertEqual(frame_indices(100, 4, 'uniform'), [0, 33, 66, 99])
        self.assertEqual(frame_indices(2, 4, 'uniform'), [0, 1])

    def test_video_group_selection_is_label_independent_and_reproducible(self):
        rows = [{'video_id': str(v), 'question_type': ('CW', 'TN', 'DL')[v % 3],
                 'question_id': f'{v}_{q}', 'label': 'A'} for v in range(20) for q in range(v % 5 + 1)]
        selected, info = select_video_groups(rows, 25, 123)
        other, _ = select_video_groups([{**q, 'label': 'E'} for q in rows], 25, 123)
        self.assertEqual([q['question_id'] for q in selected], [q['question_id'] for q in other])
        videos = {q['video_id'] for q in selected}
        self.assertEqual(selected, [q for q in rows if q['video_id'] in videos])
        self.assertEqual(set(info['video_stratum_quotas']), {'CW', 'TN', 'DL'})

    def test_wrong_video_control_never_self_assigns(self):
        mapping = wrong_video_mapping(['a', 'b', 'c', 'd'], 123)
        self.assertEqual(set(mapping), set(mapping.values()))
        self.assertTrue(all(key != value for key, value in mapping.items()))

    def test_zero_and_full_loss_preserve_expected_bytes(self):
        original = bytes(range(256)) * 9
        clean, no_loss = packet_erasure(original, 0, 123)
        lost, full_loss = packet_erasure(original, 1, 123)
        self.assertEqual(clean, original)
        self.assertEqual(lost, bytes(len(original)))
        self.assertEqual(no_loss['missing_packets'], 0)
        self.assertEqual(full_loss['zeroed_bytes'], len(original))
        self.assertEqual(full_loss['total_packets'], 3)

    def test_paired_rates_have_nested_erasure_and_partial_packet_is_bounded(self):
        original = b'\xff' * 4000
        low, _ = packet_erasure(original, .2, 123, 16)
        high, _ = packet_erasure(original, .8, 123, 16)
        self.assertEqual(len(high), len(original))
        self.assertTrue(all(a != 0 or b == 0 for a, b in zip(low, high)))


if __name__ == '__main__':
    unittest.main()
