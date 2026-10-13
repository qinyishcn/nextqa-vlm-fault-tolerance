"""Fixed temporal controls and paired, video-level visual feature damage."""
import hashlib
import random

import numpy as np
import torch

from nextqa_eval import frame_indices, packet_erasure
from llava_runtime import f32_bytes


def stable_seed(value):
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], 'little')


def timing_text(metadata, indices):
    timestamps = {row['frame_index']: row['time_seconds'] for row in metadata['frames']}
    duration = metadata['decoded_frame_count'] / metadata['average_fps']
    times = ', '.join(f'{timestamps[index]:.2f}s' for index in indices)
    return (f'The video lasts for {duration:.2f} seconds. {len(indices)} frames are provided '
            f'at these times: {times}. Please answer the following question about the video.')


def selected_features(features, cache_indices, indices):
    positions = {index: position for position, index in enumerate(cache_indices)}
    return features.index_select(0, torch.tensor([positions[index] for index in indices]))


def condition_input(condition, video, metadata, cache, all_metadata, load_cache, mapping, seed=20261012):
    kind = condition['kind']
    if kind == 'text':
        return None, None, {'kind': kind, 'video_id': video}
    indices = metadata['selections'][f'{condition["scheme"]}_{condition["frames"]}']
    timing = timing_text(metadata, indices)
    info = {'kind': kind, 'target_video_id': video, 'target_frame_indices': indices,
        'actual_frame_count': len(indices), 'timing_text': timing}
    if kind == 'text_timing':
        return None, timing, info
    feature = selected_features(cache['features'], cache['indices'], indices)
    info['clean_target_feature_sha256'] = hashlib.sha256(f32_bytes(feature)).hexdigest()
    info['source_video_id'], info['source_frame_indices'] = video, indices
    if kind == 'repeat':
        middle = frame_indices(metadata['decoded_frame_count'], 1, 'uniform')
        single = selected_features(cache['features'], cache['indices'], middle)
        feature = single.repeat(len(indices), 1, 1)
        info['source_frame_indices'] = middle * len(indices)
    elif kind in ['shuffle', 'reverse']:
        order = list(range(len(indices)))
        if kind == 'shuffle':
            random.Random(stable_seed(f'{seed}/{video}/shuffle')).shuffle(order)
            if order == list(range(len(indices))):
                order = order[1:] + order[:1]
        else:
            order.reverse()
        feature = feature[order]
        info['source_frame_indices'] = [indices[index] for index in order]
    elif kind == 'wrong_video':
        other = mapping[video]
        other_metadata = all_metadata[other]
        other_indices = frame_indices(other_metadata['decoded_frame_count'], len(indices), 'uniform')
        if len(other_indices) != len(indices):
            raise ValueError('Wrong-video control would change the declared frame count')
        wrong_cache = load_cache(other)
        feature = selected_features(wrong_cache['features'], wrong_cache['indices'], other_indices)
        info['source_video_id'], info['source_frame_indices'] = other, other_indices
    elif kind == 'gaussian':
        generator = torch.Generator().manual_seed(stable_seed(f'{seed}/{video}/gaussian'))
        feature = torch.randn(feature.shape, generator=generator, dtype=torch.float32).to(feature.dtype)
        info['gaussian_mean'], info['gaussian_std'] = 0, 1
    elif kind in ['zero', 'loss']:
        rate = 1 if kind == 'zero' else condition['loss_rate']
        repeat = condition.get('mask_repeat', 0)
        mask_seed = f'{seed}/{video}/mask_repeat_{repeat}'
        wire, erasure = packet_erasure(f32_bytes(feature), rate, mask_seed)
        feature = torch.from_numpy(np.frombuffer(wire, dtype='<f4').copy()).reshape(feature.shape).to(feature.dtype)
        if f32_bytes(feature) != wire:
            raise ValueError('Damaged feature wire bytes did not round-trip through BF16')
        info['erasure'] = erasure
    elif kind != 'clean':
        raise ValueError(f'Unexpected declared control: {kind}')
    info['condition_feature_sha256'] = hashlib.sha256(f32_bytes(feature)).hexdigest()
    return feature.contiguous(), timing, info
