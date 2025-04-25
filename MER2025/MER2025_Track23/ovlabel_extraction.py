import fire

import config
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

import os
import numpy as np
from toolkit.utils.read_files import *
from toolkit.utils.qwen import *
from toolkit.utils.functions import *
from my_affectgpt.evaluation.wheel import *

import config

def func_read_batch_calling_model(modelname):
    model_path = config.PATH_TO_LLM[modelname]
    llm = LLM(model=model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=512)
    return llm, tokenizer, sampling_params


## reason -> ov labels
def extract_openset_batchcalling(reason_root=None, reason_npz=None, update_npz=None, reason_csv=None, name2reason=None,
                                 store_root=None, store_npz=None, 
                                 modelname=None, llm=None, tokenizer=None, sampling_params=None):
    
    ## load model
    if (llm is None) and (tokenizer is None) and (sampling_params is None):
        model_path = config.PATH_TO_LLM[modelname]
        llm = LLM(model=model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=512)
   
    ## => name2reason
    if reason_root is not None:
        name2reason = func_get_name2reason(reason_root)
    elif reason_npz is not None:
        name2reason = np.load(reason_npz, allow_pickle=True)['name2reason'].tolist()
    elif reason_csv is not None:
        names = func_read_key_from_csv(reason_csv, 'name')
        reasons = func_read_key_from_csv(reason_csv, 'reason')
        name2reason = {}
        for (name, item) in zip(names, reasons):
            name2reason[name] = item
    elif update_npz is not None:
        name2reason = {}
        filenames = np.load(update_npz, allow_pickle=True)['filenames']
        fileitems = np.load(update_npz, allow_pickle=True)['fileitems']
        for (name, item) in zip(filenames, fileitems):
            name2reason[name] = item

    ## main process
    whole_names, whole_responses = list(name2reason.keys()), []
    batches_names = split_list_into_batch(whole_names, batchsize=8)
    for batch_names in batches_names:
        batch_reasons = [name2reason[name] for name in batch_names]
        batch_responses = reason_to_openset_qwen(llm=llm, tokenizer=tokenizer,
                                                 sampling_params=sampling_params, 
                                                 batch_reasons=batch_reasons)
        whole_responses.extend(batch_responses)
    
    ## storage
    if store_root is not None:
        if not os.path.exists(store_root):
            os.makedirs(store_root)
        # save to folder
        for (name, response) in zip(whole_names, whole_responses):
            save_path = os.path.join(store_root, f'{name}.npy')
            np.save(save_path, response)
    elif store_npz is not None:
        np.savez_compressed(store_npz,
                            filenames=whole_names,
                            fileitems=whole_responses)
    else:
        return whole_names, whole_responses


if __name__ == '__main__':

   
    # load model
    llm, tokenizer, sampling_params = func_read_batch_calling_model(modelname='Qwen25')

    # description -> ov labels
    for result_path in glob.glob('output/results-mer2025ov/*/*.npz'):

        # only process for result_path files
        modelname = result_path.split('/')[-2]
        if result_path.endswith('-openset.npz'): continue

        # reason -> openset
        openset_npz = result_path[:-4] + '-openset.npz'
        if not os.path.exists(openset_npz):
            extract_openset_batchcalling(reason_npz=result_path, store_npz=openset_npz,
                                        llm=llm, tokenizer=tokenizer, sampling_params=sampling_params)
            