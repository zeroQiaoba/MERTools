import os
import cv2  # We're using OpenCV to read video, to install !pip install opencv-python
import time
import base64
import requests
from openai import OpenAI

class GPT:
    def __init__(self, model_name):
        print (f'initialize GPT {model_name}')
        ###########################################
        # self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "<your OpenAI API key if not set as env var>"))
        ###########################################
        self.model_name = model_name

    # video -> 采样三帧并读取到 base64 中
    def extract_base64_frames(self, video_path, max_frames=3):
        video = cv2.VideoCapture(video_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps

        # 确定要抽取的帧索引
        if duration <= max_frames:
            # 小于等于max_frames秒的视频：每秒一帧
            frame_indices = [int(i * fps) for i in range(int(duration))]
        else:
            # 大于max_frames秒的视频：均匀采样max_frames帧
            frame_indices = [int(i * total_frames / max_frames) for i in range(max_frames)]

        base64Frames = []
        current_frame = 0
        next_index = 0

        while video.isOpened() and next_index < len(frame_indices):
            success, frame = video.read()
            if not success:
                break

            if current_frame == frame_indices[next_index]:
                _, buffer = cv2.imencode(".jpg", frame)
                base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
                next_index += 1

            current_frame += 1

        video.release()
        print(f"==> {len(base64Frames)} frames extracted from {video_path} (duration={duration}, fps={fps}, total_frames={total_frames}).")
        return base64Frames
    
    # audio -> 读取到 base64 中
    def extract_base64_audio(self, audio_path):
        
        with open(audio_path, "rb") as audio_file:
            wav_data = audio_file.read()
        encoded_string = base64.b64encode(wav_data).decode('utf-8')
        return encoded_string


    # sample-wise calling
    def func_calling_sample(self, audio_path, video_path, prompt, input_type, max_frames=3):

        if input_type == 'video':
            base64Frames = self.extract_base64_frames(video_path, max_frames=max_frames)
            
            response = self.client.responses.create(
                # model="gpt-4.1-mini",
                model=self.model_name,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": prompt
                            },
                            *[
                                {
                                    "type": "input_image",
                                    "image_url": f"data:image/jpeg;base64,{frame}"
                                }
                                for frame in base64Frames
                            ]
                        ]
                    }
                ],
            )
            print(response.output_text)
            return response.output_text

        elif input_type == 'audio':
            encoded_string = self.extract_base64_audio(audio_path)
    
            completion = self.client.chat.completions.create(
                # model="gpt-4o-audio-preview",
                model = self.model_name,
                modalities=["text", "audio"],
                audio={"voice": "alloy", "format": "wav"},
                messages=[
                    {
                        "role": "user",
                        "content": [
                            { 
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "input_audio",
                                "input_audio": {
                                    "data": encoded_string,
                                    "format": "wav"
                                }
                            }
                        ]
                    },
                ]
            )

        print(completion.choices[0].message)
        return completion.choices[0].message
