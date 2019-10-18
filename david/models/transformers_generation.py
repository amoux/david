#!/usr/bin/env python3
# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Conditional text generation with the auto-regressive models of the library (GPT/GPT-2/CTRL/Transformer-XL/XLNet)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import argparse
import logging

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange
from transformers import (CTRLConfig, CTRLLMHeadModel, CTRLTokenizer,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel,
                          OpenAIGPTTokenizer, TransfoXLConfig,
                          TransfoXLLMHeadModel, TransfoXLTokenizer, XLMConfig,
                          XLMTokenizer, XLMWithLMHeadModel, XLNetConfig,
                          XLNetLMHeadModel, XLNetTokenizer)

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop
ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys())
                  for conf in (GPT2Config, OpenAIGPTConfig, XLNetConfig,
                               TransfoXLConfig, XLMConfig, CTRLConfig)), ())
MODEL_CLASSES = {
    'gpt2': (GPT2LMHeadModel, GPT2Tokenizer),
    'ctrl': (CTRLLMHeadModel, CTRLTokenizer),
    'openai-gpt': (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'xlnet': (XLNetLMHeadModel, XLNetTokenizer),
    'transfo-xl': (TransfoXLLMHeadModel, TransfoXLTokenizer),
    'xlm': (XLMWithLMHeadModel, XLMTokenizer),
}

# Padding text to help Transformer-XL and XLNet with short prompts
# as proposed by Aman Rusia
# in https://github.com/rusiaaman/XLNet-gen#methodology and
# https://medium.com/@amanrusia/xlnet-speaks-comparison-to-gpt-2-ea1a4e9ba39e

PADDING_TEXT = """ In 1991, the remains of Russian Tsar Nicholas II and his
family (except for Alexei and Maria) are discovered.
The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
remainder of the story. 1883 Western Siberia,
a young Grigori Rasputin is asked by his father and a group of men to perform magic.
Rasputin has a vision and denounces one of the men as a horse thief. Although his
father initially slaps him for making such an accusation, Rasputin watches as the
man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
with people, even a bishop, begging for his blessing. <eod> </s> <eos>"""


def set_seed(seed, n_gpu):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0,
                          filter_value=-float('Inf')):
    """Filter a distribution of logits using top-k
    and/or nucleus (top-p) filtering.

    Args:
    -----

    `logits` :
        logits distribution shape (vocabulary size)

    `top_k` > 0 :
        keep only top k tokens with highest
        probability (top-k filtering).

    `top_p` > 0.0 :
        keep the top tokens with cumulative
        probability >= top_p (nucleus filtering).

    Nucleus filtering is described in Holtzman et al.
    (http://arxiv.org/abs/1904.09751)
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 1
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[
            0][..., -1, None]
        logits[indices_to_remove] = filter_value
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(
            F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = \
            sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


def sample_sequence(
    model,
    length,
    context,
    num_samples=1,
    temperature=1,
    top_k=0,
    top_p=0.0,
    repetition_penalty=1.0,
    is_xlnet=False,
    is_xlm_mlm=False,
    xlm_mask_token=None,
    xlm_lang=None,
    device='cpu',
):
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0).repeat(num_samples, 1)
    generated = context
    with torch.no_grad():
        for _ in trange(length):
            inputs = {'input_ids': generated}
            if is_xlnet:
                input_ids = torch.cat((generated, torch.zeros(
                    (1, 1), dtype=torch.long, device=device)), dim=1)
                perm_mask = torch.zeros(
                    (1, input_ids.shape[1], input_ids.shape[1]),
                    dtype=torch.float, device=device)
                perm_mask[:, :, -1] = 1.0
                target_mapping = torch.zeros(
                    (1, 1, input_ids.shape[1]),
                    dtype=torch.float, device=device)
                target_mapping[0, 0, -1] = 1.0
                inputs = {'input_ids': input_ids, 'perm_mask': perm_mask,
                          'target_mapping': target_mapping}
            if is_xlm_mlm and xlm_mask_token:
                input_ids = torch.cat((generated, torch.full(
                    (1, 1), xlm_mask_token,
                    dtype=torch.long, device=device)), dim=1)
                inputs = {'input_ids': input_ids}
            if xlm_lang is not None:
                inputs["langs"] = torch.tensor(
                    [xlm_lang] * inputs["input_ids"].shape[1],
                    device=device).view(1, -1)
            outputs = model(**inputs)
            next_token_logits = outputs[0][0, -1, :] / \
                (temperature if temperature > 0 else 1.)
            for _ in set(generated):
                next_token_logits[_] /= repetition_penalty
            filtered_logits = top_k_top_p_filtering(
                next_token_logits, top_k=top_k, top_p=top_p)
            if temperature == 0:
                next_token = torch.argmax(filtered_logits).unsqueeze(0)
            else:
                next_token = torch.multinomial(
                    F.softmax(filtered_logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
    return generated


def run_generation(
        model_type: str = None,
        model_name_or_path: str = None,
        prompt: str = "",
        padding_text: str = "",
        xlm_lang: str = "",
        length: int = 20,
        temperature: float = 1.0,
        repetition_penalty: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.9,
        no_cuda: bool = False,
        seed: int = 42,
        stop_token: str = None,
        stop_flag: str = 'quit',
        join_input2prompt: bool = False):
    """
    `model_type` : (str, default=None)
        Model type selected in the list.

    `model_name_or_path` : (str, default=None)
        Path to pre-trained model or shortcut name selected in the list:

    `xlm_lang` : (str, default="")
        Optional language when used with the XLM model.

    `temperature` : (float, default=1.0)
        Temperature of 0 implies greedy sampling.

    `repetition_penalty` : (float, default=1.0)
        Primarily useful for CTRL model; in that case, use 1.2

    `no_cuda` : (action='store_true')
        Avoid using CUDA when available.

    `seed` : (int, default=42)
        Random seed for initialization.

    `stop_token` : (str, default=None)
        Token at which text generation is stopped.

    `stop_flag` : (str, default="quit")
        The word (flag) to use to quit the session cleanly.
    """
    device = torch.device('cuda' if torch.cuda.is_available()
                          and not no_cuda else 'cpu')
    n_gpu = torch.cuda.device_count()
    set_seed(seed=seed, n_gpu=n_gpu)
    model_type = model_type.lower()

    model_class, tokenizer_class = MODEL_CLASSES[model_type]
    tokenizer = tokenizer_class.from_pretrained(model_name_or_path)
    model = model_class.from_pretrained(model_name_or_path)
    model.to(device)
    model.eval()

    if length < 0 and model.config.max_position_embeddings > 0:
        length = model.config.max_position_embeddings
    elif 0 < model.config.max_position_embeddings < length:
        length = model.config.max_position_embeddings
    elif length < 0:
        length = MAX_LENGTH

    logger.info(f'starting: < {model_type} >\n')
    print(f'enter < {stop_flag} > to stop the session.\n')

    if model_type in ["ctrl"]:
        if temperature > 0.7:
            logger.info(
                'CTRL typically works better with lower temperatures (and lower top_k).')

    while True:
        xlm_lang = None
        if model_type in ["xlm"]
        and hasattr(tokenizer, 'lang2id')
        and hasattr(model.config, 'use_lang_emb')
        and model.config.use_lang_emb:
            if xlm_lang:
                language = xlm_lang
            else:
                language = None
                while language not in tokenizer.lang2id.keys():
                    language = input(
                        "Using XLM. Select language in " + str(
                            list(tokenizer.lang2id.keys())) + " >>> ")
            xlm_lang = tokenizer.lang2id[language]

        is_xlm_mlm = model_type in ["xlm"] and 'mlm' in model_name_or_path
        if is_xlm_mlm:
            xlm_mask_token = tokenizer.mask_token_id
        else:
            xlm_mask_token = None

        raw_text = prompt if prompt else input("Model prompt >>> ")
        if raw_text.strip().lower() != stop_flag:
            if model_type in ["transfo-xl", "xlnet"]:
                raw_text = (
                    padding_text if padding_text else PADDING_TEXT) + raw_text
            context_tokens = tokenizer.encode(raw_text)

            out = sample_sequence(
                model=model,
                context=context_tokens,
                length=length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                is_xlnet=bool(model_type == "xlnet"),
                is_xlm_mlm=is_xlm_mlm,
                xlm_mask_token=xlm_mask_token,
                xlm_lang=xlm_lang,
                device=device)

            out = out[0, len(context_tokens):].tolist()
            text = tokenizer.decode(out, clean_up_tokenization_spaces=True,
                                    skip_special_tokens=True)
            text = text[: text.find(stop_token) if stop_token else None]
            if join_input2prompt:
                print(
                    f'predicted-text:\n\n{raw_text.strip()} {text.strip()}\n')
            else:
                print(f'predicted-paragraph(s):\n\n{text.strip()}\n')

            if prompt:
                break
        elif raw_text.strip().lower() == stop_flag:
            break
    return text
