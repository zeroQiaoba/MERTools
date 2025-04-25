<h3 align="center"><a href="https://arxiv.org/pdf/2501.16566" style="color:#9C276A">
Baselines for Track2 and Track3</a></h3>
<h5 align="center"> If our project helps you, please give us a star ⭐ on GitHub to support us. 🙏🙏 </h2>

<h5 align="center">


[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](LICENSE) 

</h5>

This project is mainly drawn from our previous work on OV-MER and AffectGPT: https://github.com/zeroQiaoba/AffectGPT

## 🛠️ Requirements and Installation
My Dependencies (We have not tested other envs):
* CUDA Version == 12.1

**[Environment Preparation]**
```bash
# we mainly depend on vllm2, but some baselines depend on whisperx
conda env create -f environment_vllm2.yml
conda env create -f environment_whisperx.yml
```

## 🚀 Dataset
In MER2025, we provide two training datasets, **OV-MERD** and **MER-Caption+**.
```bash
dataset # including both OV-MERD and MER-Caption+ Dataset
├── mer2025-dataset # Available at: https://huggingface.co/datasets/MERChallenge/MER2025
|   ├── video # all training data, including 132,171 samples
|   ├── audio # pre-extracted audio
|   ├── openface_face # # pre-extracted face files
|   ├── subtitle_chieng.csv # pre-extracted subtitle content
|   ├── track2_train_mercaptionplus.csv # MER-Caption+ Dataset (OV labels)
|   ├── track3_train_mercaptionplus.csv # MER-Caption+ Dataset (Description)
|   ├── track2_train_ovmerd.csv # OV-MERD Dataset (OV labels)
|   ├── track3_train_ovmerd.csv # OV-MERD Dataset (Description)
|   ├── track_all_candidates.csv # Only useful for participants in MER2025 [all test samples exist in these candidates]
```

Their dataset statistics are provided as below:
<p><img src="assert/dataset.png" width="800" "/></p>

<details open><summary>💡 Papers ✨. </summary><p>
<!--  may -->

> [**AffectGPT: A New Dataset, Model, and Benchmark for Emotion Understanding with Multimodal Large Language Models**](https://arxiv.org/abs/2501.16566) <br>
> Zheng Lian, Haoyu Chen, Lan Chen, Haiyang Sun, Licai Sun, Yong Ren, Zebang Cheng, Bin Liu, Rui Liu, Xiaojiang Peng, Jiangyan Yi, Jianhua Tao <br>

> [**OV-MER: Towards Open-Vocabulary Multimodal Emotion Recognition**](https://arxiv.org/abs/2410.01495) <br>
> Zheng Lian, Haiyang Sun, Licai Sun, Haoyu Chen, Lan Chen, Hao Gu, Zhuofan Wen, Shun Chen, Siyuan Zhang, Hailiang Yao, Bin Liu, Rui Liu, Shan Liang, Ya Li, Jiangyan Yi, Jianhua Tao <br>


</p></details>

## ✨ AffectGPT Training & Inference

### Pretrained Checkpoints
| Model Name     | Model Type |
|:----------------|:------------:|
| [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)  | Visual Encoder  |
| [chinese-hubert-large](https://huggingface.co/TencentGameMate/chinese-hubert-large)  | Audio Encoder |
| [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)  | LLM |
| [mercaptionplus_outputhybird_bestsetup_bestfusion_face_lz](https://pan.baidu.com/s/1R2q9_ZLtn6tgfUs4zX8gUw?pwd=givh)  | Training on **MERCaption+** and take pre-extracted **face** as input  |
| [mercaptionplus_outputhybird_bestsetup_bestfusion_frame_lz](https://pan.baidu.com/s/1iO-KyekHH3t7hDOVHy4ypg?pwd=yex1)  | Training on **MERCaption+** and take origin **frame** as input  |
| [ovmerd_outputhybird_bestsetup_bestfusion_face_lz](https://pan.baidu.com/s/1nsp1FUnYAXKMJ5kURGFc2A?pwd=ujsi)  | Training on **OV-MERD** and take pre-extracted **face** as input  |
| [ovmerd_outputhybird_bestsetup_bestfusion_frame_lz](https://pan.baidu.com/s/1Z01zSUAIlEoBaW5V7I1Pfg?pwd=4xen)  | Training on **OV-MERD** and take origin **frame** as input  |



### Data and Pre-trained Checkpoints Structure
```bash
[1] structure
dataset # including both OV-MERD and MER-Caption+ Dataset
├── mer2025-dataset # Available at: https://huggingface.co/datasets/MERChallenge/MER2025
|   ├── video # all training data, including 132,171 samples
|   ├── audio # pre-extracted audio
|   ├── openface_face # # pre-extracted face files
|   ├── subtitle_chieng.csv # pre-extracted subtitle content
|   ├── track2_train_mercaptionplus.csv # MER-Caption+ Dataset (OV labels)
|   ├── track3_train_mercaptionplus.csv # MER-Caption+ Dataset (Description)
|   ├── track2_train_ovmerd.csv # OV-MERD Dataset (OV labels)
|   ├── track3_train_ovmerd.csv # OV-MERD Dataset (Description)
|   ├── track_all_candidates.csv # Only useful for participants in MER2025 [all test samples exist in these candidates]

MER2025_Track23
├── models # Available at: https://pan.baidu.com/s/1IvC4H7Xt1AzMFocGMBBbHQ?pwd=hzf9
│   ├── chinese-hubert-large # audio encoders
│   ├── clip-vit-large-patch14 # video encoders
│   ├── Qwen2.5-7B-Instruct # LLM

[2] Please change xxx in config.py to your own path.
```

### Training
```bash
# model1: Training on MERCaptionPlus (face)
CUDA_VISIBLE_DEVICES=0 python -u train.py 
--cfg-path=train_configs/mercaptionplus_outputhybird_bestsetup_bestfusion_face_lz.yaml

# model2: Training on MERCaptionPlus (frame)
CUDA_VISIBLE_DEVICES=0 python -u train.py 
--cfg-path=train_configs/mercaptionplus_outputhybird_bestsetup_bestfusion_frame_lz.yaml

# model3: Training on OV-MERD (face)
CUDA_VISIBLE_DEVICES=0 python -u train.py 
--cfg-path=train_configs/ovmerd_outputhybird_bestsetup_bestfusion_face_lz.yaml

# model4: Training on OV-MERD (frame)
CUDA_VISIBLE_DEVICES=0 python -u train.py 
--cfg-path=train_configs/ovmerd_outputhybird_bestsetup_bestfusion_frame_lz.yaml
```

### Inference Code
1. Pre-trained Checkpoints Structure

If you want to skip the above training process, we also provide pretrained weights.
```bash
MER2025_Track23
├── models # Available at: https://pan.baidu.com/s/1IvC4H7Xt1AzMFocGMBBbHQ?pwd=hzf9
│   ├── chinese-hubert-large # audio encoders
│   ├── clip-vit-large-patch14 # video encoders
│   ├── Qwen2.5-7B-Instruct # LLM
├── output
│   ├── mercaptionplus_outputhybird_bestsetup_bestfusion_face_lz # Available: https://pan.baidu.com/s/1R2q9_ZLtn6tgfUs4zX8gUw?pwd=givh
│   ├── mercaptionplus_outputhybird_bestsetup_bestfusion_frame_lz # Available: https://pan.baidu.com/s/1iO-KyekHH3t7hDOVHy4ypg?pwd=yex1
│   ├── ovmerd_outputhybird_bestsetup_bestfusion_face_lz # Available: https://pan.baidu.com/s/1nsp1FUnYAXKMJ5kURGFc2A?pwd=ujsi
│   ├── ovmerd_outputhybird_bestsetup_bestfusion_frame_lz # Available: https://pan.baidu.com/s/1Z01zSUAIlEoBaW5V7I1Pfg?pwd=4xen
```

2. Inference => generate OV labels (**MER2025-Track2**)
save to *./output/results-mer2025ov*
```bash
# case1: Training on MERCaptionPlus (face), test for MER2025-Track2
CUDA_VISIBLE_DEVICES=0 python -u inference_hybird.py --zeroshot --dataset='MER2025OV' 
--cfg-path=train_configs/mercaptionplus_outputhybird_bestsetup_bestfusion_face_lz.yaml 
--options "inference.test_epoch=35"

# case2: Training on MERCaptionPlus (frame), test for MER2025-Track2
CUDA_VISIBLE_DEVICES=0 python -u inference_hybird.py --zeroshot --dataset='MER2025OV' 
--cfg-path=train_configs/mercaptionplus_outputhybird_bestsetup_bestfusion_frame_lz.yaml 
--options "inference.test_epoch=30" 

# case3: Training on OV-MERD (face), test for MER2025-Track2
CUDA_VISIBLE_DEVICES=0 python -u inference_hybird.py --zeroshot --dataset='MER2025OV' 
--cfg-path=train_configs/ovmerd_outputhybird_bestsetup_bestfusion_face_lz.yaml 
--options "inference.test_epoch=40" 

# case4: Training on OV-MERD (frame), test for MER2025-Track2
CUDA_VISIBLE_DEVICES=0 python -u inference_hybird.py --zeroshot --dataset='MER2025OV' 
--cfg-path=train_configs/ovmerd_outputhybird_bestsetup_bestfusion_frame_lz.yaml 
--options "inference.test_epoch=45" 
```

3. Inference => generate Descriptions (**MER2025-Track3**)
save to *./output/results-description-mer2025ov*
```bash
# case1: Training on MERCaptionPlus (face), test for MER2025-Track2
CUDA_VISIBLE_DEVICES=0 python -u inference_hybird.py --zeroshot --dataset='MER2025OV' 
--outside_user_message="Please infer the person's emotional state and provide your reasoning process."
--cfg-path=train_configs/mercaptionplus_outputhybird_bestsetup_bestfusion_face_lz.yaml 
--options "inference.test_epoch=35" "inference.base_root=output/results-description"

# case2: Training on MERCaptionPlus (frame), test for MER2025-Track2
CUDA_VISIBLE_DEVICES=0 python -u inference_hybird.py --zeroshot --dataset='MER2025OV' 
--outside_user_message="Please infer the person's emotional state and provide your reasoning process."
--cfg-path=train_configs/mercaptionplus_outputhybird_bestsetup_bestfusion_frame_lz.yaml 
--options "inference.test_epoch=30" "inference.base_root=output/results-description"

# case3: Training on OV-MERD (face), test for MER2025-Track2
CUDA_VISIBLE_DEVICES=0 python -u inference_hybird.py --zeroshot --dataset='MER2025OV' 
--outside_user_message="Please infer the person's emotional state and provide your reasoning process."
--cfg-path=train_configs/ovmerd_outputhybird_bestsetup_bestfusion_face_lz.yaml 
--options "inference.test_epoch=40" "inference.base_root=output/results-description"

# case4: Training on OV-MERD (frame), test for MER2025-Track2
CUDA_VISIBLE_DEVICES=0 python -u inference_hybird.py --zeroshot --dataset='MER2025OV' 
--outside_user_message="Please infer the person's emotional state and provide your reasoning process."
--cfg-path=train_configs/ovmerd_outputhybird_bestsetup_bestfusion_frame_lz.yaml 
--options "inference.test_epoch=45" "inference.base_root=output/results-description"
```


## 🗝️ Zero-shot Baselines
### Data and Pre-trained Checkpoints Structure
```bash
dataset # including both OV-MERD and MER-Caption+ Dataset
├── mer2025-dataset # Available at: https://huggingface.co/datasets/MERChallenge/MER2025
|   ├── video # all training data, including 132,171 samples
|   ├── audio # pre-extracted audio
|   ├── openface_face # # pre-extracted face files
|   ├── subtitle_chieng.csv # pre-extracted subtitle content
|   ├── track2_train_mercaptionplus.csv # MER-Caption+ Dataset (OV labels)
|   ├── track3_train_mercaptionplus.csv # MER-Caption+ Dataset (Description)
|   ├── track2_train_ovmerd.csv # OV-MERD Dataset (OV labels)
|   ├── track3_train_ovmerd.csv # OV-MERD Dataset (Description)
|   ├── track_all_candidates.csv # Only useful for participants in MER2025 [all test samples exist in these candidates]

MER2025_Track23
├── models # We put all pre-trained baselines weights to: https://pan.baidu.com/s/1KHL1oGCtvqr8IMNWDWxH3Q?pwd=djjw
│   ├── bert-base-uncased
│   ├── Chat-UniVi
│   ├── clip-vit-large-patch14
│   ├── LanguageBind_Image
│   ├── ...
```

### Inference Code
1. Generate Descriptions (**MER2025-Track3**) 
save to *./output/results-mer2025ov*
```bash
# vllm2
conda activate vllm2
cd Qwen-Audio
CUDA_VISIBLE_DEVICES=0 python main-audio.py --subtitle_flag='subtitle'   --dataset='MER2025OV'
cd SALMONN
CUDA_VISIBLE_DEVICES=0 python main-audio.py --subtitle_flag='subtitle'   --dataset='MER2025OV'
cd Video-ChatGPT
CUDA_VISIBLE_DEVICES=0 python main-video.py --subtitle_flag='subtitle'   --dataset='MER2025OV'
cd Chat-UniVi
CUDA_VISIBLE_DEVICES=0 python main-video.py --subtitle_flag='subtitle'   --dataset='MER2025OV'
cd mPLUG-Owl
CUDA_VISIBLE_DEVICES=0 python main-video.py --subtitle_flag='subtitle'   --dataset='MER2025OV'
cd Otter
CUDA_VISIBLE_DEVICES=0 python main-video.py --subtitle_flag='subtitle'   --dataset='MER2025OV'
cd VideoChat
CUDA_VISIBLE_DEVICES=0 python main-video.py --subtitle_flag='subtitle'   --dataset='MER2025OV'
cd VideoChat2
CUDA_VISIBLE_DEVICES=0 python main-video.py --subtitle_flag='subtitle'   --dataset='MER2025OV'

# whisperx
conda activate whisperx
cd LLaMA-VID
CUDA_VISIBLE_DEVICES=0 python main-video.py --subtitle_flag='subtitle'   --dataset='MER2025OV'
cd Video-LLaVA
CUDA_VISIBLE_DEVICES=0 python main-video.py --subtitle_flag='subtitle'   --dataset='MER2025OV'
```

2. Extract OV labels (**MER2025-Track2**)
```bash
# We provide a demo code to extract OV labels from the above emotion descriptions
python ovlabel_extraction.py
```


## 👍 Track2 Evaluation Code
```bash
# (description -> ov labels) + gt labels => score
python evaluation.py
```

Please follow the **-openset.npz* format. The Codalab evaluation system only accept **-openset.npz* format submission.



## 📑 Citation

If you find AffectGPT useful for your research and applications, please cite using this BibTeX:
```bibtex
# MER-Caption dataset, MER-Caption+ dataset, AffectGPT Framework
@article{lian2025affectgpt,
  title={AffectGPT: A New Dataset, Model, and Benchmark for Emotion Understanding with Multimodal Large Language Models},
  author={Lian, Zheng and Chen, Haoyu and Chen, Lan and Sun, Haiyang and Sun, Licai and Ren, Yong and Cheng, Zebang and Liu, Bin and Liu, Rui and Peng, Xiaojiang and others},
  journal={arXiv preprint arXiv:2501.16566},
  year={2025}
}

# OV-MERD dataset
@article{lian2024open,
  title={Open-vocabulary Multimodal Emotion Recognition: Dataset, Metric, and Benchmark},
  author={Lian, Zheng and Sun, Haiyang and Sun, Licai and Chen, Lan and Chen, Haoyu and Gu, Hao and Wen, Zhuofan and Chen, Shun and Zhang, Siyuan and Yao, Hailiang and others},
  journal={arXiv preprint arXiv:2410.01495},
  year={2024}
}

# EMER task
@article{lian2023explainable,
  title={Explainable Multimodal Emotion Recognition},
  author={Lian, Zheng and Sun, Haiyang and Sun, Licai and Gu, Hao and Wen, Zhuofan and Zhang, Siyuan and Chen, Shun and Xu, Mingyu and Xu, Ke and Chen, Kang and others},
  journal={arXiv preprint arXiv:2306.15401},
  year={2023}
}

# MER2023 Dataset
@inproceedings{lian2023mer,
  title={Mer 2023: Multi-label learning, modality robustness, and semi-supervised learning},
  author={Lian, Zheng and Sun, Haiyang and Sun, Licai and Chen, Kang and Xu, Mngyu and Wang, Kexin and Xu, Ke and He, Yu and Li, Ying and Zhao, Jinming and others},
  booktitle={Proceedings of the 31st ACM international conference on multimedia},
  pages={9610--9614},
  year={2023}
}

# MER2024 Dataset
@inproceedings{lian2024mer,
  title={Mer 2024: Semi-supervised learning, noise robustness, and open-vocabulary multimodal emotion recognition},
  author={Lian, Zheng and Sun, Haiyang and Sun, Licai and Wen, Zhuofan and Zhang, Siyuan and Chen, Shun and Gu, Hao and Zhao, Jinming and Ma, Ziyang and Chen, Xie and others},
  booktitle={Proceedings of the 2nd International Workshop on Multimodal and Responsible Affective Computing},
  pages={41--48},
  year={2024}
}
```

## 🔒 License

This project is released under the Apache 2.0 license as found in the LICENSE file.
The service is a research preview intended for **non-commercial use ONLY**. Please get in touch with us if you find any potential violations.
