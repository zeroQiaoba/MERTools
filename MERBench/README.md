<h3 align="center"><a  href="https://arxiv.org/pdf/2401.03429"  style="color:#9C276A">
MERBench: A Unified Evaluation Benchmark for Multimodal Emotion Recognition</a></h3>
<h5 align="center"> If our project helps you, please give us a star ⭐ on GitHub to support us. 🙏🙏 </h2>

<h5 align="center">

[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](../MER2025/MER2025_Track23/LICENSE) 


## 🛠️ Requirements and Installation
My Dependencies (We have not tested other envs):
* CUDA Version == 10.2
* Python == 3.8
* pytorch ==1 .8.0
* torchvision == 0.9.0
* fairseq == 0.10.1
* transformers==4.5.1
* pandas == 1.2.5
* wenetruntime
* paddlespeech == 1.4.1

**[Environment Preparation]**
```bash
conda env create -f environment.yml
```


## 🗝️ Build ./tools folder
```bash
## for face extractor (OpenFace-win)
https://drive.google.com/file/d/1-O8epcTDYCrRUU_mtXgjrS3OWA4HTp0-/view?usp=share_link  -> tools/openface_win_x64
## for visual feature extraction
https://drive.google.com/file/d/1DZVtpHWXuCmkEtwYJrTRZZBUGaKuA6N7/view?usp=share_link ->  tools/ferplus
https://drive.google.com/file/d/1wT2h5sz22SaEL4YTBwTIB3WoL4HUvg5B/view?usp=share_link ->  tools/manet
https://drive.google.com/file/d/1-U5rC8TGSPAW_ILGqoyI2uPSi2R0BNhz/view?usp=share_link ->  tools/msceleb

## for audio extraction
https://www.johnvansickle.com/ffmpeg/old-releases ->  tools/ffmpeg-4.4.1-i686-static
## for acoustic acoustic features
https://drive.google.com/file/d/1I2M5ErdPGMKrbtlSkSBQV17pQ3YD1CUC/view?usp=share_link ->  tools/opensmile-2.3.0
https://drive.google.com/file/d/1Q5BpDrZo9j_GDvCQSN006BHEuaGmGBWO/view?usp=share_link ->  tools/vggish

## huggingface for multimodal feature extracion
## We take chinese-hubert-base for example, all pre-trained models are downloaded to tools/transformers. The links for different feature extractos involved in MERBench, please refer to Table18 in our paper.
https://huggingface.co/TencentGameMate/chinese-hubert-base    -> tools/transformers/chinese-hubert-base
```

## 👍 Dataset Preprocessing
(1) You should download the raw datasets.

(2) We provide the code for dataset preprocessing.
```bash
# please refer to toolkit/proprocess for more details
see toolkit/proprocess/mer2023.py 
see toolkit/proprocess/sims.py
see toolkit/proprocess/simsv2.py
see toolkit/proprocess/cmumosi.py
see toolkit/proprocess/cmumosei.py
see toolkit/proprocess/meld.py
see toolkit/proprocess/iemocap.py
```

(3) Feature extractions

Please refer to *run.sh* for more details.

You can choose feature_level in ['UTTERANCE', 'FRAME'] to extract utterance-level or frame-level features.

You can choose '--dataset' in ['MER2023', 'IEMOCAPSix', 'CMUMOSI', 'CMUMOSEI', 'SIMS', 'MELD', 'SIMSv2'] to extract features for different datasets.

```bash
# visual features
1. extract face using openface
cd feature_extraction/visual
python extract_openface.py --dataset=MER2023 --type=videoOne

2. extract visual features
python -u extract_vision_huggingface.py --dataset=MER2023 --feature_level='UTTERANCE' --model_name='clip-vit-large-patch14'           --gpu=0    
python -u extract_vision_huggingface.py --dataset=MER2023 --feature_level='FRAME' --model_name='clip-vit-large-patch14'           --gpu=0    

# lexical features
python extract_text_huggingface.py --dataset='MER2023' --feature_level='UTTERANCE' --model_name='Baichuan-13B-Base'                     --gpu=0  
python extract_text_huggingface.py --dataset='MER2023' --feature_level='FRAME' --model_name='Baichuan-13B-Base'                     --gpu=0  

# acoustic features
1. extract 16kHZ audio from videos
python toolkit/utils/functions.py func_split_audio_from_video_16k 'dataset/sims-process/video' 'dataset/sims-process/audio'

2. extract acoustic features
python -u extract_audio_huggingface.py     --dataset='MER2023' --feature_level='UTTERANCE' --model_name='chinese-hubert-large'     --gpu=0
python -u extract_audio_huggingface.py     --dataset='MER2023' --feature_level='FRAME' --model_name='chinese-hubert-large'     --gpu=0
```

For convenience, we provide processed labels and features in *./dataset* folder.

Since features are relatively large, we upload them into Baidu Cloud Disk:
```bash
store path: ./dataset/mer2023-dataset-process   link: https://pan.baidu.com/s/1l2yrWG3wXHjdRljAk32fPQ         password: uds2 
store path: ./dataset/simsv2-process 			link: https://pan.baidu.com/s/1oJ4BP9F4s2c_JCxYVVy1UA         password: caw3 
store path: ./dataset/sims-process   			link: https://pan.baidu.com/s/1Sxfphq4IaY2K0F1Om2wNeQ         password: 60te 
store path: ./dataset/cmumosei-process  		link: https://pan.baidu.com/s/1GwTdrGM7dPIAm5o89XyaAg         password: 4fed 
store path: ./dataset/meld-process   			link: https://pan.baidu.com/s/13o7hJceXRApNsyvBO62FTQ         password: 6wje 
store path: ./dataset/iemocap-process   		link: https://pan.baidu.com/s/1k8VZBGVTs53DPF5XcvVYGQ         password: xepq 
store path: ./dataset/cmumosi-process   		link: https://pan.baidu.com/s/1RZHtDXjZsuHWnqhfwIMyFg         password: qnj5 
```



## 🚀 Baseline

### Unimodal Baseline
1. You can choose '--dataset' in ['MER2023', 'IEMOCAPSix', 'IEMOCAPFour', 'CMUMOSI', 'CMUMOSEI', 'SIMS', 'MELD', 'SIMSv2']

2. You can also change the feature names, we take three unimodal features for example.

3. By default, we randomly select hyper-parameters during training. Therefore, please run each command line 50 times, choose the best hyper-parameters, run 6 times and calculate the average result.

```bash
python -u main-release.py --model='attention' --feat_type='utt' --dataset='MER2023' --audio_feature='chinese-hubert-large-UTT' --text_feature='chinese-hubert-large-UTT' --video_feature='chinese-hubert-large-UTT' --gpu=0
python -u main-release.py --model='attention' --feat_type='utt' --dataset='MER2023' --audio_feature='clip-vit-large-patch14-UTT' --text_feature='clip-vit-large-patch14-UTT' --video_feature='clip-vit-large-patch14-UTT' --gpu=0
python -u main-release.py --model='attention' --feat_type='utt' --dataset='MER2023' --audio_feature='Baichuan-13B-Base-UTT' --text_feature='Baichuan-13B-Base-UTT' --video_feature='Baichuan-13B-Base-UTT' --gpu=0
```


### Multimodal Benchmark
We provide 5 utterance-level fusion algorithms and 5 frame-level fusion algorithms.

```bash
## for utt-level fusion
python -u main-release.py --model='attention'   --feat_type='utt'         --dataset='MER2023' --audio_feature='chinese-hubert-large-UTT' --text_feature='Baichuan-13B-Base-UTT' --video_feature='clip-vit-large-patch14-UTT' --gpu=0
python -u main-release.py --model='lmf'         --feat_type='utt'         --dataset='MER2023' --audio_feature='chinese-hubert-large-UTT' --text_feature='Baichuan-13B-Base-UTT' --video_feature='clip-vit-large-patch14-UTT' --gpu=0
python -u main-release.py --model='misa'        --feat_type='utt'         --dataset='MER2023' --audio_feature='chinese-hubert-large-UTT' --text_feature='Baichuan-13B-Base-UTT' --video_feature='clip-vit-large-patch14-UTT' --gpu=0
python -u main-release.py --model='mmim'        --feat_type='utt'         --dataset='MER2023' --audio_feature='chinese-hubert-large-UTT' --text_feature='Baichuan-13B-Base-UTT' --video_feature='clip-vit-large-patch14-UTT' --gpu=0
python -u main-release.py --model='tfn'         --feat_type='utt'         --dataset='MER2023' --audio_feature='chinese-hubert-large-UTT' --text_feature='Baichuan-13B-Base-UTT' --video_feature='clip-vit-large-patch14-UTT' --gpu=0

## for frm_align fusion
python -u main-release.py --model='mult'        --feat_type='frm_align'   --dataset='MER2023' --audio_feature='chinese-hubert-large-FRA' --text_feature='Baichuan-13B-Base-FRA' --video_feature='clip-vit-large-patch14-FRA' --gpu=0
python -u main-release.py --model='mfn'         --feat_type='frm_align'   --dataset='MER2023' --audio_feature='chinese-hubert-large-FRA' --text_feature='Baichuan-13B-Base-FRA' --video_feature='clip-vit-large-patch14-FRA' --gpu=0
python -u main-release.py --model='graph_mfn'   --feat_type='frm_align'   --dataset='MER2023' --audio_feature='chinese-hubert-large-FRA' --text_feature='Baichuan-13B-Base-FRA' --video_feature='clip-vit-large-patch14-FRA' --gpu=0
python -u main-release.py --model='mfm'         --feat_type='frm_align'   --dataset='MER2023' --audio_feature='chinese-hubert-large-FRA' --text_feature='Baichuan-13B-Base-FRA' --video_feature='clip-vit-large-patch14-FRA' --gpu=0
python -u main-release.py --model='mctn'        --feat_type='frm_align'   --dataset='MER2023' --audio_feature='chinese-hubert-large-FRA' --text_feature='Baichuan-13B-Base-FRA' --video_feature='clip-vit-large-patch14-FRA' --gpu=0
```

### Cross-corpus Benchmark
We provide both unimodal and multimodal cross-corpus benchmarks:

Please change **--train_dataset** and **--test_dataset** for cross-corpus settings.

```bash
## test for sentiment strength, we take SIMS -> CMUMOSI for example
python -u main-release.py --model=attention --feat_type='utt' --train_dataset='SIMS' --test_dataset='CMUMOSI'  --audio_feature=Baichuan-13B-Base-UTT    --text_feature=Baichuan-13B-Base-UTT --video_feature=Baichuan-13B-Base-UTT      --gpu=0
python -u main-release.py --model=attention --feat_type='utt' --train_dataset='SIMS' --test_dataset='CMUMOSI'  --audio_feature=chinese-hubert-large-UTT --text_feature=Baichuan-13B-Base-UTT --video_feature=clip-vit-large-patch14-UTT --gpu=0

## test for discrete labels, we take MER2023 -> MELD for example
python -u main-release.py --model=attention --feat_type='utt' --train_dataset='MER2023' --test_dataset='MELD'  --audio_feature=Baichuan-13B-Base-UTT    --text_feature=Baichuan-13B-Base-UTT --video_feature=Baichuan-13B-Base-UTT      --gpu=0
python -u main-release.py --model=attention --feat_type='utt' --train_dataset='MER2023' --test_dataset='MELD'  --audio_feature=chinese-hubert-large-UTT --text_feature=Baichuan-13B-Base-UTT --video_feature=clip-vit-large-patch14-UTT --gpu=0
```

## 🔒 License

This project is released under the Apache 2.0 license as found in the LICENSE file.
The service is a research preview intended for **non-commercial use ONLY**. Please get in touch with us if you find any potential violations.


## 📑 Citation

If you find MERBench useful for your research and applications, please cite using this BibTeX:
```bibtex
@article{lian2024merbench,
  title={Merbench: A unified evaluation benchmark for multimodal emotion recognition},
  author={Lian, Zheng and Sun, Licai and Ren, Yong and Gu, Hao and Sun, Haiyang and Chen, Lan and Liu, Bin and Tao, Jianhua},
  journal={arXiv preprint arXiv:2401.03429},
  year={2024}
}
```