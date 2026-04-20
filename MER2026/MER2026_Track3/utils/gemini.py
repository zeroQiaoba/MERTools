
from utils.common import *
import base64
import os
import time
from google import genai
from google.genai import types


class GEMINI:
    def __init__(self, model_name):
        print (f'initialize Gemini: {model_name}')
        client = genai.Client(api_key="AIzaSyDNAlEkXSCzSURQxocqL7_KSwfoXONOTIM")
        self.client = client
        self.model_name = model_name


    # sample-wise calling
    def func_calling_sample(self, audio_path, video_path, prompt, input_type):
        
        time.sleep(4)

        # Only for videos of size <20Mb
        video_bytes = open(video_path, 'rb').read()

        response = self.client.models.generate_content( # 测试不同版本下的模型结果
            model = self.model_name,
            contents=types.Content(
                parts=[
                    types.Part(
                        inline_data=types.Blob(data=video_bytes, mime_type='video/mp4'),
                        video_metadata=types.VideoMetadata(fps=1)
                    ),
                    types.Part(text=prompt)
                ]
            )
        )
        print(response.text)

        return response.text

