<h3 align="center"><a href="https://arxiv.org/abs/2401.03429" style="color:#9C276A">
Baselines for MER2025 Track1</a></h3>
<h5 align="center"> If our project helps you, please give us a star ⭐ on GitHub to support us. 🙏🙏 </h2>

<h5 align="center">


[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](LICENSE) 

</h5>

This project is mainly drawn from our previous work on MER2023, MER2024, and MERBench: https://github.com/zeroQiaoba/MERTools

## 🛠️ Requirements and Installation
My Dependencies (We have not tested other envs):
* CUDA Version == 12.1

**[Environment Preparation]**
```bash
conda env create -f environment_vllm2.yml
conda activate vllm2
```

## 🚀 Dataset

### Raw Dataset
We provide the training set of **MER2025-Track1**, which contains **7,369** samples. Participants need to predict emotions of all samples in *track_all_candidates.csv*.
```bash
dataset
├── mer2025-dataset # Available at: https://huggingface.co/datasets/MERChallenge/MER2025
|   ├── video # all training data, including 132,171 samples
|   ├── audio # pre-extracted audio
|   ├── openface_face # # pre-extracted face files
|   ├── subtitle_chieng.csv # pre-extracted subtitle content
|   ├── track1_train_disdim.csv # training set for MER2025-Track1
|   ├── track_all_candidates.csv # Only useful for participants in MER2025 [all test samples exist in these candidates]
```

### Dataset Preprocessing
**Note:** Since there are no test labels, we set all test labels to neutral to ensure the code runs successfully.
```bash
[1] Please change xxx in config.py to your own path.

[2] Generate *dataset/mer2025-dataset-process*
python toolkit/preprocess/mer2025.py

[3] Data structure
dataset
├── mer2025-dataset # (Raw Dataset) Available at: https://huggingface.co/datasets/MERChallenge/MER2025
|   ├── video # all training data, including 132,171 samples
|   ├── audio # pre-extracted audio
|   ├── openface_face # # pre-extracted face files
|   ├── subtitle_chieng.csv # pre-extracted subtitle content
|   ├── track1_train_disdim.csv # training set for MER2025-Track1
|   ├── track_all_candidates.csv # Only useful for participants in MER2025 [all test samples exist in these candidates]

├── **mer2025-dataset-process** # (Preprocess dataset)
|   ├── video # all training data, including 27,369 samples
|   ├── audio # pre-extracted audio
|   ├── openface_face # # pre-extracted face files
|   ├── track1_subtitle_chieng.csv # pre-extracted subtitle content
|   ├── track1_label_6way.npz # label file
```


</p></details>

## ✨ Training & Inference

### Prepare Tools
All necessary tools are avaliable at: https://pan.baidu.com/s/13dw2qQOBxgWTr0mhEseqEQ?pwd=mn6w. Please download and put them into *./tools*.
```bash
MER2025_Track1
├── tools
│   ├── emonet
│   ├── ferplus
│   ├── ...
│   ├── transformers
│   │   ├── bloom-7b1
│   │   ├── chinese-hubert-base
│   │   ├── ...
```


### Feature Extraction
You can choose *--feature_level* in ['UTTERANCE', 'FRAME'] to extract utterance-level or frame-level features.
```bash
[1] visual feature extraction
cd feature_extraction/visual
python -u extract_manet_embedding.py    --dataset=MER2025 --feature_level='UTTERANCE'                                       --gpu=0                                                
python -u extract_emonet_embedding.py   --dataset=MER2025 --feature_level='UTTERANCE'                                       --gpu=0                                                
python -u extract_ferplus_embedding.py  --dataset=MER2025 --feature_level='UTTERANCE' --model='resnet50_ferplus_dag'        --gpu=0                  
python -u extract_ferplus_embedding.py  --dataset=MER2025 --feature_level='UTTERANCE' --model='senet50_ferplus_dag'         --gpu=0                 
python -u extract_vision_huggingface.py --dataset=MER2025 --feature_level='UTTERANCE' --model_name='clip-vit-base-patch32'  --gpu=0            
python -u extract_vision_huggingface.py --dataset=MER2025 --feature_level='UTTERANCE' --model_name='clip-vit-large-patch14' --gpu=0           
python -u extract_vision_huggingface.py --dataset=MER2025 --feature_level='UTTERANCE' --model_name='videomae-base'          --gpu=0                   
python -u extract_vision_huggingface.py --dataset=MER2025 --feature_level='UTTERANCE' --model_name='videomae-large'         --gpu=0                   
python -u extract_vision_huggingface.py --dataset=MER2025 --feature_level='UTTERANCE' --model_name='dinov2-large'           --gpu=0                    

[2] audio feature extraction
cd feature_extraction/audio
python -u extract_audio_huggingface.py --dataset=MER2025 --feature_level='UTTERANCE' --model_name='chinese-hubert-base'     --gpu=0
python -u extract_audio_huggingface.py --dataset=MER2025 --feature_level='UTTERANCE' --model_name='chinese-hubert-large'    --gpu=0
python -u extract_audio_huggingface.py --dataset=MER2025 --feature_level='UTTERANCE' --model_name='chinese-wav2vec2-base'   --gpu=0
python -u extract_audio_huggingface.py --dataset=MER2025 --feature_level='UTTERANCE' --model_name='chinese-wav2vec2-large'  --gpu=0
python -u extract_audio_huggingface.py --dataset=MER2025 --feature_level='UTTERANCE' --model_name='wavlm-base'              --gpu=0
python -u extract_audio_huggingface.py --dataset=MER2025 --feature_level='UTTERANCE' --model_name='whisper-large-v2'        --gpu=0

[3] lexical feature extraction
cd feature_extraction/text
python extract_text_huggingface.py --dataset=MER2025 --feature_level='UTTERANCE' --model_name='chinese-roberta-wwm-ext'       --gpu=0
python extract_text_huggingface.py --dataset=MER2025 --feature_level='UTTERANCE' --model_name='chinese-roberta-wwm-ext-large' --gpu=0
python extract_text_huggingface.py --dataset=MER2025 --feature_level='UTTERANCE' --model_name='chinese-macbert-base'          --gpu=0
python extract_text_huggingface.py --dataset=MER2025 --feature_level='UTTERANCE' --model_name='chinese-macbert-large'         --gpu=0
python extract_text_huggingface.py --dataset=MER2025 --feature_level='UTTERANCE' --model_name='bloom-7b1'                     --gpu=0

[4] tool: check feature extraction progress
python toolkit/utils/functions.py feature_extraction_progress MER2025
```

### Unimodal Baselines
```bash
# We take *chinese-hubert-base-UTT* as an example.
# We will randomly select hyperparameters during training.
python -u main-release.py --model='attention' --feat_type='utt' --dataset=MER2025 --audio_feature=chinese-hubert-base-UTT --text_feature=chinese-hubert-base-UTT --video_feature=chinese-hubert-base-UTT --gpu=0

# We provide merged cmds for all features and each run 50 times. 
bash run-unimodal.sh

# Note:​​ Since test labels are not provided, the Train$Val score in ./saved-unimodal/result
# is accurate, but the test score is not. To obtain reliable test scores, 
# please submit your predictions for the test set to CodaLab.
```


### Multimodal Baselines

Adjust *AUDIO_RANK_LOW2HIGH  / TEXT_RANK_LOW2HIGH / IMAGR_RANK_LOW2HIGH*  in *toolkit/globals.py* based on unimodal results.

```bash
# fusion_topn: select topn features for each modality
# fusion_modality: report bimodal or trimodal results
python -u main-release.py --model=attention_topn --feat_type='utt' --dataset=MER2025 --fusion_topn=1 --fusion_modality='AVT' --gpu=0

# We provide merged cmds for all features and each run 50 times.
bash run-multimodal.sh

# Note:​​ Since test labels are not provided, the Train$Val score in ./saved-multitop-others/result
# is accurate, but the test score is not. To obtain reliable test scores, 
# please submit your predictions for the test set to CodaLab.
```


## 👍 Track1 Evaluation Code

1. Generate submission file
*xxx.npz* is generated in *main-release.py*, we need to convert *xxx.npz* to *./submission.csv* file. 
**Note:** Codalab evaluation system currently only accepts *./submission.csv* files.
```bash
python submission.py generate_submission "./saved-unimodal/result/test1_xxx.npz" "./submission.csv"
```

2. F1-Score calcuation
```bash
# The score calcuation code on Codalab
python evaluation.py score_calculation "../dataset/mer2025-dataset/track1_test_dis.csv" "./submission.csv"
```


## 📑 Citation

If you find this project useful for your research and applications, please cite using this BibTeX:
```bibtex
## MERBench
@article{lian2024merbench,
  title={Merbench: A unified evaluation benchmark for multimodal emotion recognition},
  author={Lian, Zheng and Sun, Licai and Ren, Yong and Gu, Hao and Sun, Haiyang and Chen, Lan and Liu, Bin and Tao, Jianhua},
  journal={arXiv preprint arXiv:2401.03429},
  year={2024}
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
