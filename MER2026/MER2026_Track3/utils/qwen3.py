import os
import math
import time
import glob
import tqdm
import random
import torch
import argparse
import numpy as np
import pandas as pd
import soundfile as sf
from transformers import AutoTokenizer

from utils.common import *

from vllm import LLM, SamplingParams


class QWEN3:
    def __init__(self, model_path):
        print (f'initial qwen3 model: {model_path}')

        llm = LLM(model=model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        # sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=512)
        # sampling_params = SamplingParams(max_new_tokens=32768)
        # sampling_params = SamplingParams()
        sampling_params = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=32768)

        self.llm = llm
        self.tokenizer = tokenizer
        self.sampling_params = sampling_params


    def func_postprocess_qwen(self, response):
        response = response.strip()
        if response.startswith("输入"):   response = response[len("输入"):]
        if response.startswith("输出"):   response = response[len("输出"):]
        if response.startswith("翻译"):   response = response[len("翻译"):]
        if response.startswith("让我们来翻译一下："): response = response[len("让我们来翻译一下："):]
        if response.startswith("output"): response = response[len("output"):]
        if response.startswith("Output"): response = response[len("Output"):]
        if response.startswith("input"): response = response[len("input"):]
        if response.startswith("Input"): response = response[len("Input"):]
        response = response.strip()
        if response.startswith(":"):  response = response[len(":"):]
        if response.startswith("："): response = response[len("："):]
        response = response.strip()
        response = response.replace('\n', '') # remove \n
        response = response.strip()
        return response


    def get_completion_qwen_bacth(self, prompt_list):
        
        assert isinstance(prompt_list, list)

        message_batch = []
        for prompt in prompt_list:
            message_batch.append([{"role": "user", "content": prompt}])

        text_batch = self.tokenizer.apply_chat_template(
            message_batch,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
        )

        outputs = self.llm.generate(text_batch, self.sampling_params)
        
        # => batch_responses
        batch_responses = []
        for output in outputs:
            prompt = output.prompt
            response = output.outputs[0].text
            response = response.split('</think>')[-1]
            response = self.func_postprocess_qwen(response)
            batch_responses.append(response)
            print(f"Prompt: {prompt} \n Response: {response}")
        return batch_responses

