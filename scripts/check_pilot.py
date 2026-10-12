"""Accept real checkpoint loading, feature transfer and generation parity first."""
import argparse
import collections
import gc
import hashlib
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
from transformers import Qwen2ForCausalLM

from llava_runtime import LlavaRuntime, tensor_hash
from prefix_cache import last_position_logits
from study_controls import timing_text
from run_study import canonical_hash, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['source', 'model', 'assets', 'output', 'plan', 'manifest', 'frame-audit']:
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    sys.path.insert(0, args.source)
    assets, output = Path(args.assets), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    plan, manifest, audit = [json.loads(Path(path).read_text()) for path in [args.plan, args.manifest, args.frame_audit]]
    if canonical_hash(manifest) != plan['manifest_sha256']:
        raise ValueError('Manifest no longer matches the frozen plan')
    metadata = min(audit['video_metadata'], key=lambda row: row['video_id'])
    questions = [q for q in manifest['questions'] if q['video_id'] == metadata['video_id']][:3]
    runtime = LlavaRuntime(args.model)
    save_json(output / 'runtime_provenance.json', runtime.provenance())
    tests = []

    # Real full checkpoint, normal official generate, no memory optimization.
    indices = metadata['selections']['native_center_1']
    index_rows = {row['frame_index']: row for row in metadata['frames']}
    paths = [assets / 'frames' / metadata['video_id'] / index_rows[index]['file'] for index in indices]
    timing = timing_text(metadata, indices)
    plain, plain_tokens = runtime.answer_reference(questions[0], timing, paths)
    with last_position_logits(runtime.model):
        slim, slim_tokens = runtime.answer_reference(questions[0], timing, paths)
    if plain_tokens != slim_tokens:
        raise ValueError('Last-position output head changed real-model generation')
    tests.append({'test': 'last_position_head_real_model', 'frames': 1,
                  'raw_response': plain, 'generated_token_ids': plain_tokens, 'status': 'passed'})
    print('Pilot: real checkpoint output-head parity passed', flush=True)

    with last_position_logits(runtime.model):
        for frames in [1, 4]:
            indices = metadata['selections'][f'native_center_{frames}']
            paths = [assets / 'frames' / metadata['video_id'] / index_rows[index]['file'] for index in indices]
            timing = timing_text(metadata, indices)
            captured = {}
            original_pool = runtime.model.get_2dPool

            def capture_pool(features, stride=2):
                pooled = original_pool(features, stride)
                captured['pooled'] = pooled.detach().cpu()
                return pooled

            def capture_embeddings(module, positional, kwargs):
                embeddings = kwargs.get('inputs_embeds')
                if embeddings is not None and embeddings.shape[1] > 1 and 'embedding_sha256' not in captured:
                    captured['embedding_sha256'] = tensor_hash(embeddings)

            handle = runtime.model.get_model().register_forward_pre_hook(capture_embeddings, with_kwargs=True)
            try:
                with patch.object(runtime.model, 'get_2dPool', new=capture_pool):
                    reference, reference_tokens = runtime.answer_reference(questions[0], timing, paths)
            finally:
                handle.remove()
            pooled = captured['pooled']
            ids, _ = runtime.prompt_ids(questions[0], timing, visual=True)
            received, consumption = runtime.prepare(ids, pooled)
            if tensor_hash(received) != captured['embedding_sha256']:
                raise ValueError('Lossless feature split changed actual LM input embeddings')
            split_tokens = Qwen2ForCausalLM.generate(runtime.model, inputs_embeds=received,
                do_sample=False, repetition_penalty=1.0, num_beams=1, max_new_tokens=8)
            if split_tokens[0].tolist() != reference_tokens:
                raise ValueError('Lossless feature split changed generation')
            tests.append({'test': 'official_unsplit_vs_lossless_feature_split', 'frames': frames,
                'status': 'passed', 'raw_response': reference, 'generated_token_ids': reference_tokens,
                'actual_input_embedding_sha256': captured['embedding_sha256'],
                'received_feature_sha256': consumption['received_sha256']})
            print(f'Pilot: real {frames}-frame split embeddings and generation matched', flush=True)
            del received, pooled, split_tokens, captured
            gc.collect()
            torch.cuda.empty_cache()

        # Receiver batching/caching optimizations against full receiver generate.
        for frames in [1, 4, 16, 64]:
            runtime.vision_device(runtime.device)
            indices = metadata['selections'][f'uniform_{frames}']
            paths = [assets / 'frames' / metadata['video_id'] / index_rows[index]['file'] for index in indices]
            timing = timing_text(metadata, indices)
            pooled = torch.cat([runtime.encode(paths[start:start + 4]) for start in range(0, len(paths), 4)])
            runtime.vision_device('cpu')
            references = []
            if frames <= 16:
                for question in questions:
                    ids, _ = runtime.prompt_ids(question, timing, visual=True)
                    embeddings, _ = runtime.prepare(ids, pooled)
                    tokens = Qwen2ForCausalLM.generate(runtime.model, inputs_embeds=embeddings,
                        do_sample=False, repetition_penalty=1.0, num_beams=1, max_new_tokens=8)
                    references.append(tokens[0].tolist())
                    del embeddings, tokens
                    torch.cuda.empty_cache()
            prefix, suffixes, receiver = runtime.build_prefix(questions, timing, pooled)
            answers = []
            for index, suffix in enumerate(suffixes):
                raw, tokens = runtime.answer_cached(suffix, prefix)
                if references and tokens != references[index]:
                    raise ValueError('Shared prefix or chunked prefill changed real-model generation')
                answers.append({'raw_response': raw, 'generated_token_ids': tokens})
            tests.append({'test': 'shared_prefix_chunked_prefill_receiver', 'frames': frames,
                'status': 'passed', 'full_receiver_reference_checked': frames <= 16,
                'actual_receiver': receiver, 'answers': answers})
            print(f'Pilot: {frames}-frame cached receiver ran; full-reference checked={frames <= 16}', flush=True)
            del prefix, suffixes, pooled
            gc.collect()
            torch.cuda.empty_cache()
    result = {'status': 'passed', 'plan_sha256': canonical_hash(plan), 'video_id': metadata['video_id'],
        'question_ids': [q['question_id'] for q in questions], 'tests': tests,
        'limitation': 'Full receiver generation parity checked at 1/4/16 frames. '
            '64-frame input uses the tested chunked path due to 8GB GPU memory; '
            'no full unchunked 64-frame reference is claimed.',
        'pilot_answers_not_formal_scores': True, 'runtime': runtime.provenance()}
    save_json(output / 'pilot.json', result)
    print('Pilot acceptance passed; formal scoring may begin', flush=True)


if __name__ == '__main__':
    main()
