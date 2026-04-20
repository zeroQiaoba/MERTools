import sys
sys.path.append('../')

from argparse import ArgumentParser
import copy

from pllava.utils.easydict import EasyDict
from pllava.tasks.eval.model_utils import load_pllava
from pllava.tasks.eval.eval_utils import (
    ChatPllava,
    conv_plain_v1,
    Conversation,
    conv_templates
)
from pllava.tasks.eval.demo import pllava_theme

SYSTEM="""You are a powerful Video Magic ChatBot, a large vision-language assistant. 
You are able to understand the video content that the user provides and assist the user in a video-language related task.
The user might provide you with the video and maybe some extra noisy information to help you out or ask you a question. Make use of the information in a proper way to be competent for the job.
### INSTRUCTIONS:
1. Follow the user's instruction.
2. Be critical yet believe in yourself.
"""

INIT_CONVERSATION: Conversation = conv_plain_v1.copy()
INIT_CONVERSATION = conv_templates['plain']

class PLLAVA:
    def __init__(self, model_path):
        print ('initial pllava model')

        self.num_frames=16
        self.lora_alpha=4
        self.use_lora=True
        self.use_multi_gpus = False

        model, processor = load_pllava(
            model_path,
            self.num_frames, 
            use_lora=self.use_lora, 
            weight_dir=model_path, 
            lora_alpha=self.lora_alpha, 
            use_multi_gpus=self.use_multi_gpus)
        if not self.use_multi_gpus:
            model = model.to('cuda')
        chat = ChatPllava(model, processor)

        self.chat = chat


    def upload_img(self, gr_video, chat_state=None, img_list=None):
        chat_state = INIT_CONVERSATION.copy() if chat_state is None else chat_state
        img_list = [] if img_list is None else img_list
        
        msg, img_list, chat_state = self.chat.upload_video(gr_video, chat_state, img_list, None)
        return (
            chat_state,
            img_list,
        )
       
    
    def gradio_ask(self, user_message, chat_state, system):
        chat_state = self.chat.ask(user_message, chat_state, system)
        return chat_state

    def gradio_answer(self, chat_state, img_list, num_beams, temperature):
        llm_message, llm_message_token, chat_state = self.chat.answer(conv=chat_state, img_list=img_list, max_new_tokens=200, num_beams=num_beams, temperature=temperature)
        llm_message = llm_message.replace("<s>", "").strip() # handle <s>
        print(llm_message)
        return llm_message


    # sample-wise calling
    def func_calling_sample(self, audio_path, video_path, prompt, input_type):
        
        num_beams = 1
        temperature = 1.0
        system_string = SYSTEM

        chat_state = None
        img_list = None

        # text_input = 'Describe the background, characters and the actions in the provided video.'

        # read input
        chat_state, img_list = self.upload_img(video_path, chat_state)
        chat_state = self.gradio_ask(prompt, chat_state, system_string)
        response = self.gradio_answer(chat_state, img_list, num_beams, temperature)

        print(response)
        
        return response

