<h3 align="center"><a  href="https://dl.acm.org/doi/pdf/10.1145/3581783.3612836"  style="color:#9C276A">
MER 2023: Multi-label Learning, Modality Robustness, and Semi-Supervised Learning</a></h3>
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
To download the dataset, please fill out an EULA on Hugging Face: https://huggingface.co/datasets/MERChallenge/MER2023. It requires participants to use this dataset only for academic research and not to edit or upload samples to the Internet. The official website is http://merchallenge.cn/.


> [**MER 2023: Multi-label Learning, Modality Robustness, and Semi-Supervised Learning**](https://dl.acm.org/doi/pdf/10.1145/3581783.3612836) <br>
> Zheng Lian, Haiyang Sun, Licai Sun, Jinming Zhao, Ye Liu, Bin Liu, Jiangyan Yi, Meng Wang, Erik Cambria, Guoying Zhao, Björn W. Schuller, Jianhua Tao <br>


</p></details>


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
https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_large.pt  -> tools/wav2vec

## download wenet model and move to tools/wenet
visit "https://github.com/wenet-e2e/wenet/blob/main/docs/pretrained_models.en.md" fill the request link and download
"https://wenet-1256283475.cos.ap-shanghai.myqcloud.com/models/wenetspeech/wenetspeech_u2pp_conformer_libtorch.tar.gz"

## huggingface for multimodal feature extracion
https://huggingface.co/TencentGameMate/chinese-hubert-base    -> tools/transformers/chinese-hubert-base
https://huggingface.co/TencentGameMate/chinese-hubert-large   -> tools/transformers/chinese-hubert-large
https://huggingface.co/TencentGameMate/chinese-wav2vec2-base  -> tools/transformers/chinese-wav2vec2-base
https://huggingface.co/TencentGameMate/chinese-wav2vec2-large -> tools/transformers/chinese-wav2vec2-large
https://huggingface.co/bert-base-chinese -> tools/transformers/bert-base-chinese
https://huggingface.co/hfl/chinese-roberta-wwm-ext -> tools/transformers/chinese-roberta-wwm-ext
https://huggingface.co/hfl/chinese-roberta-wwm-ext-large -> tools/transformers/chinese-roberta-wwm-ext-large
https://huggingface.co/WENGSYX/Deberta-Chinese-Large -> tools/transformers/deberta-chinese-large
https://huggingface.co/hfl/chinese-electra-180g-small-discriminator -> tools/transformers/chinese-electra-180g-small
https://huggingface.co/hfl/chinese-electra-180g-base-discriminator -> tools/transformers/chinese-electra-180g-base
https://huggingface.co/hfl/chinese-electra-180g-large-discriminator -> tools/transformers/chinese-electra-180g-large
https://huggingface.co/hfl/chinese-xlnet-base -> tools/transformers/chinese-xlnet-base
https://huggingface.co/hfl/chinese-macbert-base -> tools/transformers/chinese-macbert-base
https://huggingface.co/hfl/chinese-macbert-large -> tools/transformers/chinese-macbert-large
https://huggingface.co/IDEA-CCNL/Taiyi-CLIP-Roberta-102M-Chinese -> tools/transformers/taiyi-clip-roberta-chinese
https://huggingface.co/IDEA-CCNL/Wenzhong2.0-GPT2-3.5B-chinese -> tools/transformers/wenzhong2-gpt2-chinese
https://huggingface.co/clue/albert_chinese_tiny -> tools/transformers/albert_chinese_tiny
https://huggingface.co/clue/albert_chinese_small -> tools/transformers/albert_chinese_small

## for audio corruption (only use the speech subset)
https://www.openslr.org/17/ -> tools/musan
```

## 👍 Baseline
```bash
# step1: dataset preprocess
python main-baseline.py normalize_dataset_format --data_root='./dataset-release' --save_root='./dataset-process'

# step2: multimodal feature extraction (see run_release.sh for more examples)
## you can choose feature_level in ['UTTERANCE', 'FRAME'] 
python -u extract_wav2vec_embedding.py --dataset='MER2023' --feature_level='UTTERANCE' --gpu=0

# step3: training unimodal and multimodal classifiers (see run_release.sh for more examples)
## You can choose test_sets in [test1, test2, test3]. Currently, we only provide test3. [test1, test2] will provide in July 1, 2023. In this code, test1, test2, test3 refer to MER-MULTI, MER-NOISE, and MER-SEMI, respectively.
## unimodal
python -u main-release.py --dataset='MER2023' --test_sets='test3' --audio_feature='manet_UTT' --text_feature='manet_UTT' --video_feature='manet_UTT' --lr=1e-3 --gpu=0
## multimodal
python -u main-release.py --dataset='MER2023' --model_type='attention' --test_sets='test3' --audio_feature='chinese-hubert-large-UTT' --text_feature='chinese-macbert-large-4-UTT' --video_feature='chinese-macbert-large-4-UTT' --lr=1e-3 --gpu=0
```

## 🚀 Other Examples
For other datasets, please refer to *run_release.sh*

For data corruption strategies in MER-NOISE, submission format, and evaluation metrics, please refer to：
```bash
## data corruption methods: corrupt videos in video_root, and save to save_root
python main-corrupt.py main_mixture_multiprocess(video_root, save_root)

## submission format
step1: "write_to_csv_pred(name2preds, pred_path)" in main-release.py
step2: submit "pred_path"

## evaluation metrics
for [test1, test2] => "report_results_on_test1_test2(label_path, pred_path)" in main-release.py 
for [test3]        => "report_results_on_test3(label_path, pred_path)"       in main-release.py 
```

## 🔒 License

This project is released under the Apache 2.0 license as found in the LICENSE file.
The service is a research preview intended for **non-commercial use ONLY**. Please get in touch with us if you find any potential violations.


## 📑 Citation

If you find MER2023 useful for your research and applications, please cite using this BibTeX:
```bibtex
# MER2023 Dataset
@inproceedings{lian2023mer,
  title={Mer 2023: Multi-label learning, modality robustness, and semi-supervised learning},
  author={Lian, Zheng and Sun, Haiyang and Sun, Licai and Chen, Kang and Xu, Mngyu and Wang, Kexin and Xu, Ke and He, Yu and Li, Ying and Zhao, Jinming and others},
  booktitle={Proceedings of the 31st ACM international conference on multimedia},
  pages={9610--9614},
  year={2023}
}
```