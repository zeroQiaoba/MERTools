<h3 align="center"><a  href="https://dl.acm.org/doi/abs/10.1145/3689092.3689959"  style="color:#9C276A">
MER 2024: Semi-Supervised Learning, Noise Robustness, and Open-Vocabulary Multimodal Emotion Recognition</a></h3>
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


## :earth_americas: Dataset
To download the dataset, please fill out an EULA in https://drive.google.com/file/d/1cXNfKHyJzVXg_7kWSf_nVKtsxIZVa517/view?usp=sharing, and send it to lianzheng2016@ia.ac.cn. It requires participants to use this dataset only for academic research and not to edit or upload samples to the Internet.


> [**MER 2024: Semi-Supervised Learning, Noise Robustness, and Open-Vocabulary Multimodal Emotion Recognition**](https://dl.acm.org/doi/abs/10.1145/3689092.3689959) <br>
> Zheng Lian, Haiyang Sun, Licai Sun, Zhuofan Wen, Siyuan Zhang, Shun Chen, Hao Gu, Jinming Zhao, Ziyang Ma, Xie Chen, Jiangyan Yi, Rui Liu, Kele Xu, Bin Liu, Erik Cambria, Guoying Zhao, Björn W. Schuller, Jianhua Tao <br>




</p></details>


## 🗝️ Build ./tools folder
```bash
## for face extractor (OpenFace-win)
https://drive.google.com/file/d/1-O8epcTDYCrRUU_mtXgjrS3OWA4HTp0-/view?usp=share_link  -> tools/openface_win_x64
## for visual feature extraction
https://drive.google.com/file/d/1DZVtpHWXuCmkEtwYJrTRZZBUGaKuA6N7/view?usp=share_link ->  tools/ferplus
https://drive.google.com/file/d/1wT2h5sz22SaEL4YTBwTIB3WoL4HUvg5B/view?usp=share_link ->  tools/manet
https://drive.google.com/file/d/1-U5rC8TGSPAW_ILGqoyI2uPSi2R0BNhz/view?usp=share_link ->  tools/msceleb
https://drive.google.com/file/d/1sg-sI8QgFJwfPfWC1fQkVXDVvoQiGbxY/view?usp=sharing    ->  tools/videomae-base-VoxCeleb2/checkpoint-99.pth
https://drive.google.com/file/d/14TpZHsdICfce36rw6j3oB_hdbpdX6DHK/view?usp=sharing    ->  tools/videomae-base-K400-mer2023/checkpoint-299.pth

## for audio extraction
https://www.johnvansickle.com/ffmpeg/old-releases ->  tools/ffmpeg-4.4.1-i686-static
## for acoustic acoustic features
https://drive.google.com/file/d/1I2M5ErdPGMKrbtlSkSBQV17pQ3YD1CUC/view?usp=share_link ->  tools/opensmile-2.3.0
https://drive.google.com/file/d/1Q5BpDrZo9j_GDvCQSN006BHEuaGmGBWO/view?usp=share_link ->  tools/vggish

## ASR: download wenet model and move to tools/wenet => you can use other ASR toolkits
visit "https://github.com/wenet-e2e/wenet/blob/main/docs/pretrained_models.en.md" fill the request link and download
"https://wenet-1256283475.cos.ap-shanghai.myqcloud.com/models/wenetspeech/wenetspeech_u2pp_conformer_libtorch.tar.gz"

## huggingface for multimodal feature extracion
## We take chinese-hubert-base for example, all pre-trained models are downloaded to tools/transformers. The links for different feature extractos involved in MERBench, please refer to Table18 in our paper.
https://huggingface.co/TencentGameMate/chinese-hubert-base    -> tools/transformers/chinese-hubert-base
```

## 👍 Dataset Preprocessing
1. Download the dataset and put it into ./mer2024-dataset

2. We provide the code for dataset preprocessing.
```bash
# see toolkit/preprocess/mer2024.py
```

3. Feature extractions. Please refer to run-mer2024.sh for more details. You can choose feature_level in ['UTTERANCE', 'FRAME'] to extract utterance-level or frame-level features.


## 🚀 Baseline

### Unimodal Baseline
1. You can also change the feature names, we take *eGeMAPS_UTT* for example.

2. By default, we randomly select hyper-parameters during training. Therefore, please run each command line 50 times and choose the best hyper-parameters.
```bash
python -u main-release.py --model='attention' --feat_type='utt' --dataset=MER2024 --audio_feature=eGeMAPS_UTT --text_feature=eGeMAPS_UTT --video_feature=eGeMAPS_UTT --gpu=0
```

### Multimodal Benchmark
1. Adjust *AUDIO_RANK_LOW2HIGH / TEXT_RANK_LOW2HIGH / IMAGR_RANK_LOW2HIGH* in *toolkit/globals.py*

2. Training => automatic selection top-n features for each modality
```bash
## [fusion_topn 1~6; fusion_modality in ['AV', 'AT', 'VT', 'AVT']]
python -u main-release.py --model=attention_topn --feat_type='utt' --dataset=MER2024 --fusion_topn=5 --fusion_modality='AVT' --gpu=0
```

### Others
1. We provide the code about how to generate noise samples => [for MER-NOISE]
```bash
## for audio corruption (only use the speech subset)
https://www.openslr.org/17/ -> tools/musan

## noise sample generation
python main-noise.py main_mixture_multiprocess 
            --video_root='/share/home/lianzheng/chinese-mer-2023/dataset/mer2024-dataset-process/video-test2' 
            --save_root='/share/home/lianzheng/chinese-mer-2023/dataset/mer2024-dataset-process/video-test2-noise' 
```

2. We provide the code about how to evaluate OV emotion recognition performance => [for MER-OV]
```bash
## step1: 
set your OpenAI key to candidate_keys = ["your key"] => see toolkit/utils/chatgpt.py

## step2: calculate set-level accuracy and recall for two example files in ./ov_store
# gt_csv:   checked open-vocabulary labels
# pred_csv: predicted open-vocabulary labels
python main-ov.py main_metric
            --gt_csv='ov_store/check-openset.csv'
            --pred_csv='ov_store/predict-openset.csv'

## If you use our pre-extracted ./ov_store/openset-synonym.zip, you can get the following results:
set level accuracy: 0.581777108433735; recall: 0.4977911646586345
avg score: 0.5397841365461847
```

## 🔒 License

This project is released under the Apache 2.0 license as found in the LICENSE file.
The service is a research preview intended for **non-commercial use ONLY**. Please get in touch with us if you find any potential violations.

## 📑 Citation

If you find MER2024 useful for your research and applications, please cite using this BibTeX:
```bibtex
@inproceedings{lian2024mer,
  title={Mer 2024: Semi-supervised learning, noise robustness, and open-vocabulary multimodal emotion recognition},
  author={Lian, Zheng and Sun, Haiyang and Sun, Licai and Wen, Zhuofan and Zhang, Siyuan and Chen, Shun and Gu, Hao and Zhao, Jinming and Ma, Ziyang and Chen, Xie and others},
  booktitle={Proceedings of the 2nd International Workshop on Multimodal and Responsible Affective Computing},
  pages={41--48},
  year={2024}
}
```