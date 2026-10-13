"""Official LLaVA-Video Qwen2 inference with an audited pooled-feature boundary."""
import contextlib
import copy
import hashlib
import importlib.metadata
import json
import types
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from nextqa_eval import question_prompt
from prefix_cache import generate_from_prefix, warm_prefix


def f32_bytes(tensor):
    return tensor.detach().float().cpu().contiguous().numpy().astype('<f4', copy=False).tobytes()


def tensor_hash(tensor):
    return hashlib.sha256(f32_bytes(tensor)).hexdigest()


def common_prefix(sequences):
    length = min(len(row) for row in sequences)
    for index in range(length):
        if any(row[index] != sequences[0][index] for row in sequences[1:]):
            return index
    return length


class LlavaRuntime:
    def __init__(self, checkpoint):
        import bitsandbytes as bnb
        from transformers import BitsAndBytesConfig
        from llava.model.builder import load_pretrained_model
        from llava.model.language_model.llava_qwen import LlavaQwenForCausalLM
        from llava.model.multimodal_encoder.siglip_encoder import SigLipVisionModel, SigLipVisionConfig

        self.device, self.dtype = torch.device('cuda:0'), torch.bfloat16
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError('This protocol requires CUDA BF16 support')
        self.loading_info = None
        original_loader = LlavaQwenForCausalLM.from_pretrained

        def checked_loader(*args, **kwargs):
            model, info = original_loader(*args, output_loading_info=True, **kwargs)
            self.loading_info = info
            if any(info.get(key) for key in ['missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs']):
                raise RuntimeError(f'Checkpoint parameters were not loaded exactly: {info}')
            return model

        # All tuned vision weights are present in the full checkpoint. Construct
        # the official module on meta devices, then let the outer loader fill it;
        # avoid downloading unrelated Google initialization weights to overwrite.
        def empty_vision(cls, name, **kwargs):
            if name != 'google/siglip-so400m-patch14-384':
                raise ValueError('Unexpected vision initialization source')
            return SigLipVisionModel(SigLipVisionConfig())

        quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=self.dtype,
            llm_int8_skip_modules=['vision_tower', 'mm_projector', 'lm_head'])
        with patch.object(SigLipVisionModel, 'from_pretrained', new=classmethod(empty_vision)), \
                patch.object(LlavaQwenForCausalLM, 'from_pretrained', new=checked_loader):
            self.tokenizer, self.model, self.processor, self.context_length = load_pretrained_model(
                str(checkpoint), None, 'llava_qwen', torch_dtype='bfloat16', device_map='auto',
                quantization_config=quantization, attn_implementation='sdpa', local_files_only=True)
        self.model.eval()
        self.vision = self.model.get_vision_tower()
        self.vision.to(self.device, dtype=self.dtype)
        self.model.get_model().mm_projector.to(self.device, dtype=self.dtype)
        if len(self.vision.vision_tower.vision_model.encoder.layers) != 26:
            raise ValueError('Unexpected tuned SigLIP layer layout')
        self.linear4bit_count = sum(isinstance(module, bnb.nn.Linear4bit) for module in self.model.modules())
        if self.linear4bit_count != 196 or any(isinstance(module, bnb.nn.Linear4bit)
                for owner in [self.vision, self.model.get_model().mm_projector, self.model.lm_head]
                for module in owner.modules()):
            raise ValueError('Unexpected NF4 quantization scope')
        if any(parameter.device.type != 'cuda' for parameter in self.model.parameters()):
            raise ValueError('Unexpected automatic CPU offload; not the declared runtime')

    def provenance(self):
        packages = ['torch', 'torchvision', 'transformers', 'tokenizers', 'bitsandbytes',
                    'accelerate', 'numpy', 'Pillow', 'safetensors', 'open_clip_torch', 'timm']
        return {'model': 'lmms-lab/LLaVA-Video-7B-Qwen2',
            'revision': '013210b3aff822f1558b166d39c1046dd109520f',
            'quantization': 'LM linear layers: NF4 + double quantization; BF16 compute',
            'unquantized_modules': ['vision_tower', 'mm_projector', 'lm_head', 'embed_tokens'],
            'linear4bit_count': self.linear4bit_count,
            'loading_info': self.loading_info, 'context_length': self.context_length,
            'gpu': torch.cuda.get_device_name(0), 'cuda': torch.version.cuda,
            'packages': {key: importlib.metadata.version(key) for key in packages},
            'feature_boundary': 'official get_2dPool output, before add_token_per_grid',
            'feature_wire_dtype': 'little-endian float32',
            'implementation_sha256': {name: hashlib.sha256(
                (Path(__file__).parent / name).read_text(encoding='utf-8').encode()).hexdigest()
                for name in ['llava_runtime.py', 'prefix_cache.py', 'study_controls.py']},
            'generation': {'do_sample': False, 'num_beams': 1, 'repetition_penalty': 1.0,
                           'max_new_tokens': 8, 'eos_token_id': self.model.generation_config.eos_token_id}}

    def vision_device(self, device):
        self.vision.to(device=device, dtype=self.dtype)
        torch.cuda.empty_cache()

    @torch.inference_mode()
    def pixels(self, paths):
        images = []
        for path in paths:
            with Image.open(path) as image:
                images.append(image.convert('RGB'))
        return self.processor.preprocess(images, return_tensors='pt')['pixel_values'].to(
            device=self.device, dtype=self.dtype)

    @torch.inference_mode()
    def encode(self, paths):
        projected = self.model.encode_images(self.pixels(paths))
        pooled = self.model.get_2dPool(projected)
        if tuple(pooled.shape[1:]) != (196, 3584) or not torch.isfinite(pooled).all():
            raise ValueError('Unexpected or nonfinite official pooled visual features')
        return pooled.detach().cpu().contiguous()

    def prompt_ids(self, question, timing=None, visual=False):
        from llava.conversation import conv_templates
        from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
        from llava.mm_utils import tokenizer_image_token
        content = question_prompt(question)
        if timing:
            content = timing + '\n' + content
        if visual:
            content = DEFAULT_IMAGE_TOKEN + '\n' + content
        conv = copy.deepcopy(conv_templates['qwen_1_5'])
        conv.append_message(conv.roles[0], content)
        conv.append_message(conv.roles[1], None)
        text = conv.get_prompt()
        ids = tokenizer_image_token(text, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        return ids.to(self.device).unsqueeze(0), hashlib.sha256(text.encode()).hexdigest()

    @contextlib.contextmanager
    def receive_pooled(self, pooled):
        """Feed only received model features through unchanged official packing."""
        received = pooled.to(device=self.device, dtype=self.dtype)
        calls = {'encode': 0, 'pool': 0, 'received_sha256': None}

        def encoded_stub(model, images):
            calls['encode'] += 1
            if len(images) != len(received):
                raise ValueError('Receiver frame count mismatch')
            return torch.zeros((len(received), 1, received.shape[-1]), device=self.device, dtype=self.dtype)

        def received_pool(model, unused, stride=2):
            calls['pool'] += 1
            if len(unused) != len(received):
                raise ValueError('Receiver pooled layout mismatch')
            calls['received_sha256'] = tensor_hash(received)
            return received

        with patch.object(self.model, 'encode_images', new=types.MethodType(encoded_stub, self.model)), \
                patch.object(self.model, 'get_2dPool', new=types.MethodType(received_pool, self.model)):
            yield calls
        if calls['encode'] != 1 or calls['pool'] != 1:
            raise ValueError('The receiver did not consume exactly one visual feature buffer')

    @torch.inference_mode()
    def prepare(self, ids, pooled=None):
        if pooled is None:
            return self.model.get_model().embed_tokens(ids), {}
        dummy = torch.zeros((len(pooled), 3, 1, 1), device=self.device, dtype=self.dtype)
        with self.receive_pooled(pooled) as calls:
            values = self.model.prepare_inputs_labels_for_multimodal(ids, None, None, None, None,
                [dummy], modalities=['video'])
        embeddings = values[4]
        if embeddings.shape[1] != ids.shape[1] - 1 + len(pooled) * 210:
            raise ValueError('Unexpected grid-newline packing or context truncation')
        if calls['received_sha256'] != tensor_hash(pooled):
            raise ValueError('Actual received feature hash changed during dtype conversion')
        return embeddings, calls

    @torch.inference_mode()
    def build_prefix(self, questions, timing=None, pooled=None):
        sequences, prompt_hashes = [], []
        for question in questions:
            ids, digest = self.prompt_ids(question, timing, pooled is not None)
            sequences.append(ids[0].tolist())
            prompt_hashes.append(digest)
        length = min(common_prefix(sequences), min(len(row) - 1 for row in sequences))
        if length < 1:
            raise ValueError('Question suffix must remain outside the shared prefix')
        ids = torch.tensor([sequences[0][:length]], dtype=torch.long, device=self.device)
        embeddings, received = self.prepare(ids, pooled)
        if embeddings.shape[1] + max(len(row) - length for row in sequences) + 8 > self.context_length:
            raise ValueError('Benchmark context would be truncated')
        metadata = {'prefix_token_ids_sha256': hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest(),
            'prefix_embedding_sha256': tensor_hash(embeddings), 'prefix_tokens': embeddings.shape[1],
            'receiver': received, 'prompt_sha256': prompt_hashes}
        cache = warm_prefix(self.model, embeddings)
        suffixes = [torch.tensor([row[length:]], dtype=torch.long, device=self.device) for row in sequences]
        return cache, suffixes, metadata

    @torch.inference_mode()
    def answer_cached(self, suffix, cache):
        tokens = generate_from_prefix(self.model, suffix, cache)
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)[0].strip(), tokens[0].tolist()

    @torch.inference_mode()
    def answer_reference(self, question, timing=None, paths=None):
        ids, _ = self.prompt_ids(question, timing, visual=paths is not None)
        kwargs = {}
        if paths is not None:
            kwargs = {'images': [self.pixels(paths)], 'modalities': ['video']}
        tokens = self.model.generate(ids, do_sample=False, repetition_penalty=1.0,
            num_beams=1, max_new_tokens=8, use_cache=True, **kwargs)
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)[0].strip(), tokens[0].tolist()
