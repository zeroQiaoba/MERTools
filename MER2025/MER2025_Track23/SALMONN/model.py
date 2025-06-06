# Copyright (2023) Tsinghua University, Bytedance Ltd. and/or its affiliates
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


import torch
import soundfile as sf
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    WhisperFeatureExtractor,
    WhisperModel,
    LlamaForCausalLM,
    LlamaTokenizer
)
import librosa
from beats.BEATs import BEATsConfig, BEATs
from qformer.Qformer import BertConfig, BertLMHeadModel

class SALMONN(nn.Module):
    def __init__(
        self,
        ckpt,
        whisper_path,
        beats_path,
        vicuna_path,
        speech_qformer_token_num=1,
        speech_qformer_layer=2,
        lora=True,
        lora_alpha=32,
        lora_rank=8,
        lora_dropout=0.1,
        second_per_frame=0.333333,
        second_stride=0.333333,
        low_resource=False
    ):

        super().__init__()

        # whisper
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(whisper_path)
        self.speech_encoder = WhisperModel.from_pretrained(whisper_path).encoder
        self.ln_speech = nn.LayerNorm(self.speech_encoder.config.d_model)

        # beats [frozen]
        self.beats_ckpt = beats_path
        beats_checkpoint = torch.load(self.beats_ckpt, map_location='cpu')
        beats_cfg = BEATsConfig(beats_checkpoint['cfg'])
        beats = BEATs(beats_cfg)
        beats.load_state_dict(beats_checkpoint['model'])
        self.beats = beats
        self.ln_audio = nn.LayerNorm(self.beats.cfg.encoder_embed_dim)
        for name, param in self.beats.named_parameters():
            param.requires_grad = False
        self.beats.eval()

        # init speech Qformer
        self.speech_Qformer, self.speech_query_tokens = self.init_speech_Qformer(
            speech_qformer_token_num,
            self.speech_encoder.config.d_model + self.beats.cfg.encoder_embed_dim,
            speech_qformer_layer,
        )
        self.second_per_frame = second_per_frame
        self.second_stride = second_stride
        
        # vicuna => 这些组建都是视觉模型里面共用的
        if not low_resource:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                vicuna_path,
                torch_dtype=torch.float16,
            )
        else:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                vicuna_path,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map={'': 0}
            )

        # lora
        self.lora = lora
        if lora: # 对模型进行修改，在 (q,v)处增加lora旁路
            target_modules = None
            self.peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, 
                inference_mode=True, 
                r=lora_rank, 
                lora_alpha=lora_alpha, 
                lora_dropout=lora_dropout,
                target_modules=target_modules,
            )
            self.llama_model = get_peft_model(self.llama_model, self.peft_config)

        # proj [audio -> llm hidden]
        self.speech_llama_proj = nn.Linear(
            self.speech_Qformer.config.hidden_size, self.llama_model.config.hidden_size)
        
        # tokenizer
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(vicuna_path, use_fast=False)
        self.llama_tokenizer.add_special_tokens({'pad_token': '[PAD]'}) 
        self.llama_tokenizer.padding_side = "right"

        # load ckpt [这里面存储了所有trainable params, 包括 LLaMA and speech_Qformer 相关的参数进行更新]
        ckpt_dict = torch.load(ckpt, map_location=torch.device('cpu'))['model']
        self.load_state_dict(ckpt_dict, strict=False)

    def generate(
        self,
        wav_path,
        prompt,
        prompt_pattern="USER: <Speech><SpeechHere></Speech> {}\nASSISTANT:",
        device='cuda:0',
        max_length=200,
        num_beams=4,
        do_sample=True,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        temperature=1.0,
    ):
        # read wav
        wav, sr = sf.read(wav_path)
        if len(wav.shape) == 2:
            wav = wav[:, 0]
        # ------------------------------- #
        # 我剩余的内存大小有限，大于4s的就卡到4s算了
        if len(wav) > 30 * sr:
            wav = wav[: 30 * sr]
        # if len(wav) > 4 * sr: 
        #     wav = wav[: 4 * sr]
        # ------------------------------- #
        if sr != 16000:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=16000, res_type="fft")
        
        # whisper
        spectrogram = self.feature_extractor(wav, return_tensors="pt", sampling_rate=16000).input_features.to(device) # [1, 80, 3000]
        speech_embeds = self.speech_encoder(spectrogram, return_dict=True).last_hidden_state # [1, 1500, 1280]
       
        # beats
        raw_wav = torch.from_numpy(wav).to(device).unsqueeze(0)
        audio_padding_mask = torch.zeros(raw_wav.shape, device=device).bool()
        audio_embeds, _ = self.beats.extract_features(raw_wav, padding_mask=audio_padding_mask, feature_only=True) # [1, 56, 768]

        # auditory embeds => len(speech) > len(audio)
        speech_embeds = self.ln_speech(speech_embeds)
        audio_embeds = self.ln_audio(audio_embeds)
        audio_embeds = F.pad(audio_embeds, (0, 0, 0, speech_embeds.size(1) - audio_embeds.size(1)))
        speech_embeds = torch.cat([speech_embeds, audio_embeds], dim=-1) # [1, 1500, 2048]

        # split frames -> 这块应该在说 window-wise q-former
        B, T, C = speech_embeds.shape # [1, 1500, 2048]
        kernel = round(T * self.second_per_frame / 30.0) # 17
        stride = round(T * self.second_stride / 30.0)    # 17
        kernel = (1, kernel) # (1, 17)
        stride = (1, stride) # (1, 17)
        speech_embeds_tr = speech_embeds.transpose(1, 2).unsqueeze(2) # [batch=1, dim=2048, 1, seqlen=1500]
        speech_embeds_overlap = F.unfold(speech_embeds_tr, kernel_size=kernel, dilation=1, padding=0, stride=stride) # 按照1D conv的格式展开为 [batch=1, dim=2048*1*17, seqlen=1500/17]
        _, _, L = speech_embeds_overlap.shape # L=1500/17
        speech_embeds_overlap = speech_embeds_overlap.view(B, -1, kernel[1], L)    # [1, 2048, 17, 1500/17]
        speech_embeds_overlap = torch.permute(speech_embeds_overlap, [0, 3, 2, 1]) # [1, 1500/17, 17, 2048]
        speech_embeds = speech_embeds_overlap.reshape(-1, kernel[1], C) # [1500/17=88, 17, 2048] ?? 感觉跟reshape没啥区别
        speech_atts = torch.ones(speech_embeds.size()[:-1], dtype=torch.long, device=speech_embeds.device) # [88, 17]

        # Qformer
        query_tokens = self.speech_query_tokens.expand(speech_embeds.shape[0], -1, -1) # [1, 1, 768] -> [88, 1, 768]
        query_output = self.speech_Qformer.bert( 
            query_embeds=query_tokens,
            encoder_hidden_states=speech_embeds,
            encoder_attention_mask=speech_atts,
            return_dict=True,
        )
        speech_embeds = self.speech_llama_proj(query_output.last_hidden_state) # [88, 1, 768] -> [88, 1, 5120]
        speech_embeds = speech_embeds.view(B, -1, speech_embeds.size(2)).contiguous() # [1, 88, 5120]
        speech_atts = torch.ones(speech_embeds.size()[:-1], dtype=torch.long).to(speech_embeds.device) # [1, 88]

        # USER: <Speech>speech_embeds<Speech> prompt\nASSISTANT:
        embed_tokens = self.llama_model.model.model.embed_tokens if self.lora else self.llama_model.model.embed_tokens # 词典 Embedding(32000, 5120, padding_idx=0)
        prompt_left, prompts_right = prompt_pattern.format(prompt).split('<SpeechHere>') # USER: <Speech><SpeechHere></Speech> {}\nASSISTANT: -> 分成左右两半
        prompt_left_ids = self.llama_tokenizer(
            prompt_left,
            return_tensors="pt",
            add_special_tokens=False
        ).to(speech_embeds.device).input_ids
        prompt_left_embeds = embed_tokens(prompt_left_ids)
        prompt_right_ids = self.llama_tokenizer(
            prompts_right,
            return_tensors="pt",
            add_special_tokens=False
        ).to(speech_embeds.device).input_ids
        prompt_right_embeds = embed_tokens(prompt_right_ids)

        bos_embeds = self.llama_model.model.embed_tokens(
            torch.ones(
                [1, 1],
                dtype=torch.long,
                device=device,
            ) * self.llama_tokenizer.bos_token_id
        ) if not self.lora else self.llama_model.model.model.embed_tokens(
            torch.ones(
                [1, 1],
                dtype=torch.long,
                device=device,
            ) * self.llama_tokenizer.bos_token_id # <s>
        ) # bos_embeds: [1, 1, dim=5120]

        embeds = torch.cat([bos_embeds, prompt_left_embeds, speech_embeds, prompt_right_embeds], dim=1)
        atts = torch.ones(embeds.size()[:-1], dtype=torch.long).to(embeds.device)

        # generate
        '''
        on cpu [没有gpu的情况，将半精度转成全精度才能进行前馈]: self.llama_model = self.llama_model.float()
        '''
        if device == 'cpu': # cpu 不支持 amp
            self.llama_model = self.llama_model.float()
        output = self.llama_model.generate(
            inputs_embeds=embeds,
            max_length=max_length,
            num_beams=num_beams,
            do_sample=do_sample,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature,
            attention_mask=atts,
            bos_token_id=self.llama_tokenizer.bos_token_id,
            eos_token_id=self.llama_tokenizer.eos_token_id,
            pad_token_id=self.llama_tokenizer.pad_token_id
        )
        
        output_text = self.llama_tokenizer.batch_decode(output, add_special_tokens=False, skip_special_tokens=True)

        return output_text

    def init_speech_Qformer(self, num_query_token, speech_width, num_hidden_layers=2):
        encoder_config = BertConfig()
        encoder_config.num_hidden_layers = num_hidden_layers
        encoder_config.encoder_width = speech_width
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = 1
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens
