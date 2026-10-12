import unittest

try:
    import torch
    from study_controls import condition_input
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'Requires the separate inference environment')
class FeatureControlTests(unittest.TestCase):
    def setUp(self):
        self.metadata = {'decoded_frame_count': 8, 'average_fps': 30,
            'frames': [{'frame_index': i, 'time_seconds': i / 30} for i in range(8)],
            'selections': {'uniform_4': [0, 2, 5, 7]}}
        self.cache = {'features': torch.arange(8, dtype=torch.bfloat16).reshape(8, 1, 1).repeat(1, 4, 8),
                      'indices': list(range(8))}

    def control(self, kind):
        return condition_input({'kind': kind, 'frames': 4, 'scheme': 'uniform'}, 'v1', self.metadata,
            self.cache, {'v1': self.metadata, 'v2': self.metadata},
            lambda video: {'features': self.cache['features'] + 20, 'indices': list(range(8))},
            {'v1': 'v2', 'v2': 'v1'})

    def test_wrong_video_changes_content_but_preserves_target_timing(self):
        clean, timing, _ = self.control('clean')
        wrong, other_timing, info = self.control('wrong_video')
        self.assertEqual(timing, other_timing)
        self.assertTrue(torch.equal(wrong, clean + 20))
        self.assertEqual(info['source_video_id'], 'v2')

    def test_zero_features_remain_distinct_from_absent_vision(self):
        absent, text_timing, _ = self.control('text_timing')
        zero, zero_timing, info = self.control('zero')
        self.assertIsNone(absent)
        self.assertEqual(text_timing, zero_timing)
        self.assertEqual(torch.count_nonzero(zero).item(), 0)
        self.assertEqual(info['erasure']['zeroed_bytes'], info['erasure']['feature_bytes'])

    def test_repeat_and_reverse_preserve_layout_with_declared_content(self):
        repeated, _, info = self.control('repeat')
        self.assertEqual(info['source_frame_indices'], [3] * 4)
        self.assertTrue(torch.equal(repeated, torch.full_like(repeated, 3)))
        reversed_features, _, info = self.control('reverse')
        self.assertEqual(info['source_frame_indices'], [7, 5, 2, 0])
        self.assertEqual(reversed_features[:, 0, 0].tolist(), [7, 5, 2, 0])
