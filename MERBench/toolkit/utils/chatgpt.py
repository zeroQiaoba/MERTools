
import os
import cv2
import glob
import base64
import numpy as np

import openai

# avoid RPD errors
global_index = 1
candidate_keys = ["sk-xxxx", "sk-xxxx", "sk-xxxx"] # Please use your own APIs, we support multiple APIs
openai.api_key = candidate_keys[global_index]

# 单次调用
def func_get_completion(prompt, model="gpt-3.5-turbo-16k-0613"):
    try:
        messages = [{"role": "user", "content": prompt}]
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0, # this is the degree of randomness
            max_tokens=1000,
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        print ('发生错误：', e) # change key to avoid RPD
        global global_index # 修改全局变量
        global_index = (global_index + 1) % 3
        print (f'========== key index: {global_index} ==========')
        openai.api_key = candidate_keys[global_index]
        return ''

# 多次调用，避免网络异常
def get_completion(prompt, model, maxtry=5):
    response = ''
    try_number = 0
    while len(response) == 0:
        try_number += 1
        if try_number == maxtry: 
            print (f'fail for {maxtry} times')
            break
        response = func_get_completion(prompt, model)
    return response

# chatgpt输出结果后处理
def func_postprocess_chatgpt(response):
    response = response.strip()
    if response.startswith("输入"):   response = response[len("输入"):]
    if response.startswith("输出"):   response = response[len("输出"):]
    if response.startswith("翻译"):   response = response[len("翻译"):]
    if response.startswith("让我们来翻译一下："): response = response[len("让我们来翻译一下："):]
    if response.startswith("output"): response = response[len("output"):]
    if response.startswith("Output"): response = response[len("Output"):]
    response = response.strip()
    if response.startswith(":"):  response = response[len(":"):]
    if response.startswith("："): response = response[len("："):]
    response = response.strip()
    response = response.replace('\n', '') # remove \n
    response = response.strip()
    return response


# ---------------------------------------------------------------------
## convert image/video into GPT4 support version
def func_image_to_base64(image_path, grey_flag=False): # support more types
    image = cv2.imread(image_path)
    if grey_flag:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return func_opencv_to_base64(image)

def func_opencv_to_base64(image):
    _, buffer = cv2.imencode('.jpg', image)
    base64_image = base64.b64encode(buffer).decode('utf-8')
    return base64_image

# deal with text
def func_nyp_to_text(npy_path):
    text = np.load(npy_path).tolist()
    text = text.strip()
    text = text.replace('\n', '') # remove \n
    text = text.replace('\t', '') # remove \t
    text = text.strip()
    return text

# ---------------------------------------------------------------------
## Translation
# ---------------------------------------------------------------------
def get_translate_eng2chi(text, model='gpt-3.5-turbo-16k-0613'):
    if len(text) == 0:
        return ""
    
    prompt = f"""
              请将以下输入翻译为中文：
              
              输入：{text}

              输出：
              """
    response = get_completion(prompt, model)
    response = func_postprocess_chatgpt(response)
    print (text)
    print (response)
    return response


def get_translate_chi2eng(text, model='gpt-3.5-turbo-16k-0613'):
    if len(text)==0:
        return ""
    
    prompt = f"""
              请将以下输入翻译为英文：

              输入：{text}

              输出：
              """
    response = get_completion(prompt, model)
    response = func_postprocess_chatgpt(response)
    print (text)
    print (response)
    return response


if __name__ == '__main__':
    
    ## text input [test ok]
    text = 'The whether is sooooo good!!'
    get_translate_eng2chi(text, model='gpt-3.5-turbo-16k-0613')
