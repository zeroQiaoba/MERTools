# *_*coding:utf-8 *_*
import os
import time
import argparse
import numpy as np
import pandas as pd

import torch
from transformers import AutoModel, BertTokenizer, AutoTokenizer # version: 4.5.1, pip install transformers
from transformers import GPT2Tokenizer, GPT2Model, AutoModelForCausalLM

# local folder
import sys
sys.path.append('../../')
import config

##################### English #####################
BERT_BASE = 'bert-base-cased'
BERT_LARGE = 'bert-large-cased'
BERT_BASE_UNCASED = 'bert-base-uncased'
BERT_LARGE_UNCASED = 'bert-large-uncased'
ALBERT_BASE = 'albert-base-v2'
ALBERT_LARGE = 'albert-large-v2'
ALBERT_XXLARGE = 'albert-xxlarge-v2'
ROBERTA_BASE = 'roberta-base'
ROBERTA_LARGE = 'roberta-large'
ELECTRA_BASE = 'electra-base-discriminator'
ELECTRA_LARGE = 'electra-large-discriminator'
XLNET_BASE = 'xlnet-base-cased'
XLNET_LARGE = 'xlnet-large-cased'
T5_BASE = 't5-base'
T5_LARGE = 't5-large'
DEBERTA_BASE = 'deberta-base'
DEBERTA_LARGE = 'deberta-large'
DEBERTA_XLARGE = 'deberta-v2-xlarge'
DEBERTA_XXLARGE = 'deberta-v2-xxlarge'

##################### Chinese #####################
BERT_BASE_CHINESE = 'bert-base-chinese' # https://huggingface.co/bert-base-chinese
ROBERTA_BASE_CHINESE = 'chinese-roberta-wwm-ext' # https://huggingface.co/hfl/chinese-roberta-wwm-ext
ROBERTA_LARGE_CHINESE = 'chinese-roberta-wwm-ext-large' # https://huggingface.co/hfl/chinese-roberta-wwm-ext-large
DEBERTA_LARGE_CHINESE = 'deberta-chinese-large' # https://huggingface.co/WENGSYX/Deberta-Chinese-Large
ELECTRA_SMALL_CHINESE = 'chinese-electra-180g-small' # https://huggingface.co/hfl/chinese-electra-180g-small-discriminator
ELECTRA_BASE_CHINESE  = 'chinese-electra-180g-base' # https://huggingface.co/hfl/chinese-electra-180g-base-discriminator
ELECTRA_LARGE_CHINESE = 'chinese-electra-180g-large' # https://huggingface.co/hfl/chinese-electra-180g-large-discriminator
XLNET_BASE_CHINESE = 'chinese-xlnet-base' # https://huggingface.co/hfl/chinese-xlnet-base
MACBERT_BASE_CHINESE = 'chinese-macbert-base' # https://huggingface.co/hfl/chinese-macbert-base
MACBERT_LARGE_CHINESE = 'chinese-macbert-large' # https://huggingface.co/hfl/chinese-macbert-large
PERT_BASE_CHINESE = 'chinese-pert-base' # https://huggingface.co/hfl/chinese-pert-base
PERT_LARGE_CHINESE = 'chinese-pert-large' # https://huggingface.co/hfl/chinese-pert-large
LERT_SMALL_CHINESE = 'chinese-lert-small' # https://huggingface.co/hfl/chinese-lert-small
LERT_BASE_CHINESE  = 'chinese-lert-base' # https://huggingface.co/hfl/chinese-lert-base
LERT_LARGE_CHINESE = 'chinese-lert-large' # https://huggingface.co/hfl/chinese-lert-large
GPT2_CHINESE = 'gpt2-chinese-cluecorpussmall' # https://huggingface.co/uer/gpt2-chinese-cluecorpussmall
CLIP_CHINESE = 'taiyi-clip-roberta-chinese' # https://huggingface.co/IDEA-CCNL/Taiyi-CLIP-Roberta-102M-Chinese
WENZHONG_GPT2_CHINESE = 'wenzhong2-gpt2-chinese' # https://huggingface.co/IDEA-CCNL/Wenzhong2.0-GPT2-3.5B-chinese
ALBERT_TINY_CHINESE = 'albert_chinese_tiny' # https://huggingface.co/clue/albert_chinese_tiny
ALBERT_SMALL_CHINESE = 'albert_chinese_small' # https://huggingface.co/clue/albert_chinese_small
SIMBERT_BASE_CHINESE = 'simbert-base-chinese' # https://huggingface.co/WangZeJun/simbert-base-chinese

##################### Multilingual #####################
MPNET_BASE = 'paraphrase-multilingual-mpnet-base-v2' # https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2

##################### LLM #####################
LLAMA_7B  = 'llama-7b-hf' # https://huggingface.co/decapoda-research/llama-7b-hf
LLAMA_13B = 'llama-13b-hf' # https://huggingface.co/decapoda-research/llama-13b-hf
LLAMA2_7B = 'llama-2-7b' # https://huggingface.co/meta-llama/Llama-2-7b
LLAMA2_13B = 'Llama-2-13b-hf' # https://huggingface.co/NousResearch/Llama-2-13b-hf
VICUNA_7B  = 'vicuna-7b-v0' # https://huggingface.co/lmsys/vicuna-7b-delta-v0
VICUNA_13B = 'stable-vicuna-13b' # https://huggingface.co/CarperAI/stable-vicuna-13b-delta
ALPACE_13B = 'chinese-alpaca-2-13b' # https://huggingface.co/ziqingyang/chinese-alpaca-2-13b
MOSS_7B = 'moss-base-7b' # https://huggingface.co/fnlp/moss-base-7b
STABLEML_7B = 'stablelm-base-alpha-7b-v2' # https://huggingface.co/stabilityai/stablelm-base-alpha-7b-v2
BLOOM_7B = 'bloom-7b1' # https://huggingface.co/bigscience/bloom-7b1
CHATGLM2_6B = 'chatglm2-6b' # https://huggingface.co/THUDM/chatglm2-6b
# reley on pytorch=2.0 => env: videollama4 + cpu
FALCON_7B = 'falcon-7b' # https://huggingface.co/tiiuae/falcon-7b
# Baichuan: pip install transformers_stream_generator
BAICHUAN_7B = 'Baichuan-7B' # https://huggingface.co/baichuan-inc/Baichuan-7B
BAICHUAN_13B = 'Baichuan-13B-Base' # https://huggingface.co/baichuan-inc/Baichuan-13B-Base
# BAICHUAN2_7B: conda install xformers -c xformers
BAICHUAN2_7B = 'Baichuan2-7B-Base' # https://huggingface.co/baichuan-inc/Baichuan2-7B-Base
# BAICHUAN2_13B: pip install accelerate
BAICHUAN2_13B = 'Baichuan2-13B-Base' # https://huggingface.co/baichuan-inc/Baichuan2-13B-Base
OPT_13B = 'opt-13b' # https://huggingface.co/facebook/opt-13b


################################################################
# 自动删除无意义token对应的特征
def find_start_end_pos(tokenizer):
    sentence = '今天天气真好' # 句子中没有空格
    input_ids = tokenizer(sentence, return_tensors='pt')['input_ids'][0]
    start, end = None, None

    # find start, must in range [0, 1, 2]
    for start in range(0, 3, 1):
        # 因为decode有时会出现空格，因此我们显示的时候把这部分信息去掉看看
        outputs = tokenizer.decode(input_ids[start:]).replace(' ', '')
        if outputs == sentence:
            print (f'start: {start};  end: {end}')
            return start, None

        if outputs.startswith(sentence):
            break
   
    # find end, must in range [-1, -2]
    for end in range(-1, -3, -1):
        outputs = tokenizer.decode(input_ids[start:end]).replace(' ', '')
        if outputs == sentence:
            break
    
    assert tokenizer.decode(input_ids[start:end]).replace(' ', '') == sentence
    print (f'start: {start};  end: {end}')
    return start, end


# 找到 batch_pos and feature_dim
def find_batchpos_embdim(tokenizer, model, gpu):
    sentence = '今天天气真好'
    inputs = tokenizer(sentence, return_tensors='pt')
    if gpu != -1: inputs = inputs.to('cuda')

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True).hidden_states # for new version 4.5.1
        outputs = torch.stack(outputs)[[-1]].sum(dim=0) # sum => [batch, T, D=768]
        outputs = outputs.cpu().numpy() # (B, T, D) or (T, B, D)
        batch_pos = None
        if outputs.shape[0] == 1:
            batch_pos = 0
        if outputs.shape[1] == 1:
            batch_pos = 1
        assert batch_pos in [0, 1]
        feature_dim = outputs.shape[2]
    print (f'batch_pos:{batch_pos}, feature_dim:{feature_dim}')
    return batch_pos, feature_dim


# main process
def extract_embedding(model_name, trans_dir, save_dir, feature_level, gpu=-1, punc_case=None, language='chinese', model_dir=None):

    print('='*30 + f' Extracting "{model_name}" ' + '='*30)
    start_time = time.time()

    # save last four layers
    layer_ids = [-4, -3, -2, -1]

    # save_dir
    if punc_case is None and language == 'chinese' and model_dir is None:
        save_dir = os.path.join(save_dir, f'{model_name}-{feature_level[:3]}')
    elif punc_case is not None:
        save_dir = os.path.join(save_dir, f'{model_name}-punc{punc_case}-{feature_level[:3]}')
    elif language == 'english':
        save_dir = os.path.join(save_dir, f'{model_name}-langeng-{feature_level[:3]}')
    elif model_dir is not None:
        prefix_name = "-".join(model_dir.split('/')[-2:])
        save_dir = os.path.join(save_dir, f'{prefix_name}-{model_name}-{feature_level[:3]}')
    if not os.path.exists(save_dir): os.makedirs(save_dir)

    # load model and tokenizer: offline mode (load cached files) # 函数都一样，但是有些位置的参数就不好压缩
    print('Loading pre-trained tokenizer and model...')
    if model_dir is None:
        model_dir = os.path.join(config.PATH_TO_PRETRAINED_MODELS, f'transformers/{model_name}')

    if model_name in [DEBERTA_LARGE_CHINESE, ALBERT_TINY_CHINESE, ALBERT_SMALL_CHINESE]:
        model = AutoModel.from_pretrained(model_dir)
        tokenizer = BertTokenizer.from_pretrained(model_dir, use_fast=False)
    elif model_name in [WENZHONG_GPT2_CHINESE]:
        model = GPT2Model.from_pretrained(model_dir)
        tokenizer = GPT2Tokenizer.from_pretrained(model_dir, use_fast=False)
    elif model_name in [LLAMA_7B, LLAMA_13B, LLAMA2_7B, VICUNA_7B, VICUNA_13B, ALPACE_13B, OPT_13B, BLOOM_7B]:
        model = AutoModel.from_pretrained(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    elif model_name in [LLAMA2_13B]:
        model = AutoModel.from_pretrained(model_dir, use_safetensors=False)
        tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    elif model_name in [CHATGLM2_6B, MOSS_7B]:
        model = AutoModel.from_pretrained(model_dir, trust_remote_code=True)
        if model_dir in ['pretrain-guhao/merged_chatglm2_2', 'pretrain-guhao/merged_chatglm2_3']:
            tokenizer = AutoTokenizer.from_pretrained('pretrain-guhao/merged_chatglm2', use_fast=False, trust_remote_code=True)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)
    elif model_name in [BAICHUAN_7B, BAICHUAN_13B, BAICHUAN2_7B, BAICHUAN2_13B]:
        model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)
    elif model_name in [STABLEML_7B]:
        model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    else:
        model = AutoModel.from_pretrained(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    
    # 有 gpu 并且是 LLM，才会增加 half process
    if gpu != -1 and model_name in [LLAMA_7B, LLAMA_13B, LLAMA2_7B, LLAMA2_13B, VICUNA_7B, VICUNA_13B, ALPACE_13B, 
                                    OPT_13B, BLOOM_7B, CHATGLM2_6B, MOSS_7B, BAICHUAN_7B, FALCON_7B, BAICHUAN_13B, 
                                    STABLEML_7B, BAICHUAN2_7B, BAICHUAN2_13B]:
        model = model.half()

    # 有 gpu 才会放在cuda上
    if gpu != -1:
        torch.cuda.set_device(gpu)
        model.cuda()
    model.eval()

    print('Calculate embeddings...')
    start, end = find_start_end_pos(tokenizer) # only preserve [start:end+1] tokens
    batch_pos, feature_dim = find_batchpos_embdim(tokenizer, model, gpu) # find batch pos

    df = pd.read_csv(trans_dir)
    for idx, row in df.iterrows():
        name = row['name']
        # --------------------------------------------------
        if language == 'chinese':
            sentence = row['chinese'] # process on Chinese
        elif language == 'english':
            sentence = row['english']
        # --------------------------------------------------
        print(f'Processing {name} ({idx}/{len(df)})...')

        # extract embedding from sentences
        embeddings = []
        if pd.isna(sentence) == False and len(sentence) > 0:
            inputs = tokenizer(sentence, return_tensors='pt')
            if gpu != -1: inputs = inputs.to('cuda')
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True).hidden_states # for new version 4.5.1
                outputs = torch.stack(outputs)[layer_ids].sum(dim=0) # sum => [batch, T, D=768]
                outputs = outputs.cpu().numpy() # (B, T, D)
                if batch_pos == 0:
                    embeddings = outputs[0, start:end]
                elif batch_pos == 1:
                    embeddings = outputs[start:end, 0]

        # align with label timestamp and write csv file
        print (f'feature dimension: {feature_dim}')
        csv_file = os.path.join(save_dir, f'{name}.npy')
        if feature_level == 'FRAME':
            embeddings = np.array(embeddings).squeeze()
            if len(embeddings) == 0:
                embeddings = np.zeros((1, feature_dim))
            elif len(embeddings.shape) == 1:
                embeddings = embeddings[np.newaxis, :]
            np.save(csv_file, embeddings)
        else:
            embeddings = np.array(embeddings).squeeze()
            if len(embeddings) == 0:
                embeddings = np.zeros((feature_dim, ))
            elif len(embeddings.shape) == 2:
                embeddings = np.mean(embeddings, axis=0)
            np.save(csv_file, embeddings)

    end_time = time.time()
    print(f'Total {len(df)} files done! Time used ({model_name}): {end_time - start_time:.1f}s.')


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Run.')
    parser.add_argument('--dataset', type=str, help='input dataset')
    parser.add_argument('--gpu', type=int, default='1', help='gpu id')
    parser.add_argument('--model_name', type=str, help='name of pretrained model')
    parser.add_argument('--feature_level', type=str, default='UTTERANCE', choices=['UTTERANCE', 'FRAME'], help='output types')
    # ------ 临时测试标点符号对于结果的影响 ------
    parser.add_argument('--punc_case', type=str, default=None, help='test punc impact to the performance')
    # ------ 临时Language对于结果的影响 ------
    parser.add_argument('--language', type=str, default='chinese', help='used language')
    # ------ 临时测试外部接受的 model_dir [for gu hao] ------
    parser.add_argument('--model_dir', type=str, default=None, help='used user-defined model_dir')
    args = parser.parse_args()
    
    # (trans_dir, save_dir)
    if args.punc_case is None:
        trans_dir = config.PATH_TO_TRANSCRIPTIONS[args.dataset]
    else:
        assert args.punc_case in ['case1', 'case2', 'case3']
        trans_dir = config.PATH_TO_TRANSCRIPTIONS[args.dataset][:-4] + f'-{args.punc_case}.csv'
        assert os.path.exists(trans_dir)
    save_dir = config.PATH_TO_FEATURES[args.dataset]

    extract_embedding(model_name=args.model_name, 
                      trans_dir=trans_dir, 
                      save_dir=save_dir,
                      feature_level=args.feature_level,
                      gpu=args.gpu,
                      punc_case=args.punc_case,
                      language=args.language,
                      model_dir=args.model_dir)
