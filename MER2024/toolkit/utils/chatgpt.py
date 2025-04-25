
import time
import openai

global_index = 0
candidate_keys = [""]
openai.api_key = candidate_keys[global_index]

# single call
def func_get_completion(prompt, model="gpt-3.5-turbo-16k-0613"):
    try:
        messages = [{"role": "user", "content": prompt}]
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0,
            max_tokens=1000,
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        print ('Errors: ', e)
        global global_index
        global_index = (global_index + 1) % len(candidate_keys)
        print (f'========== key index: {global_index} ==========')
        openai.api_key = candidate_keys[global_index]
        return ''

# multiple calls
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

# post-process outputs
def func_postprocess_chatgpt(response):
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


# get synonym
def get_openset_synonym(gt_openset, pred_openset, sleeptime=0, model='gpt-3.5-turbo-16k-0613'):
    merge_openset = list(set(gt_openset) | set(pred_openset))
    prompt = [
                {
                    "type": "text", 
                    "text": f"Please assume the role of an expert in the field of emotions. We provide a set of emotions. \
Please group the emotions, with each group containing emotions with the same meaning. \
Directly output the results. The output format should be a list containing multiple lists. \
Input: ['Agree', 'agreement', 'Relaxed', 'acceptance', 'pleasant', 'relaxed', 'Accept', 'positive', 'Happy'] Output: [['Agree', 'agreement', 'Accept', 'acceptance'], ['Relaxed', 'relaxed'],['pleasant', 'positive', 'Happy']] \
Input: {merge_openset} Output:"
                }
            ]
    print (prompt[0]['text'])
    for item in prompt: print (item['type'])
    time.sleep(sleeptime)
    response = get_completion(prompt, model)
    response = func_postprocess_chatgpt(response)
    print (response)
    return response

