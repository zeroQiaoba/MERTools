<div align="center">

# Track1: MER-Cross

### Baselines for Interlocutor Emotion Recognition

> This project builds upon our previous work on MER2023, MER2024, MER2025, and MERBench.
> Reference repository: [MERTools](https://github.com/zeroQiaoba/MERTools)

</div>

---

## Table of Contents

- [Task Description](#-task-description)
- [Dataset](#-dataset)
- [Training & Inference](#-training--inference)

---

## 📌 Task Description

MER-Cross is a **newly introduced track** that shifts the focus from individual scenarios to **dyadic interaction** scenarios. As illustrated below, in a dyadic conversation, two characters s₁ and s₂ speak in alternating turns.

- When **s₁ is speaking**: multimodal cues (audio, text, video) are available for s₁; only **visual** cues are available for s₂ (the listener).
- Previous MER tasks focused on **s₁'s emotions** (the speaker).
- MER-Cross focuses on **s₂'s emotions** (the interlocutor / listener).

By integrating both tasks, we can capture the emotional states of **both participants** in dynamic interaction scenarios.

<div align="center">
  <img src="../icon/track1.png" alt="MER-Cross Task" width="85%">
  <p><em>MER-Cross: Unlike previous tasks that focus on isolated speakers, we aim to predict the emotions of interlocutors, capturing both speakers' emotional states in dynamic interaction scenarios.</em></p>
</div>


---

## 🚀 Dataset

### Raw Dataset

We provide 9,395 training samples with individual emotion labels. Participants need to predict emotions for all samples in `track1_track2_candidate.csv`.

```
dataset/
└── mer2026-dataset/                      # https://huggingface.co/datasets/MERChallenge/MER2026
    ├── video/                            # 132,171 samples
    ├── audio/                            # 132,171 samples
    ├── openface_face/                    # 132,171 samples
    ├── subtitle_chieng.csv               # Pre-extracted subtitles, 132,171 samples
    ├── track1_train.csv                  # Training set (same-person), 9,395 samples
    └── track1_track2_candidate.csv       # 20,000 candidate samples
```

### Dataset Preprocessing

> **Note:** Since there are no test labels, all test labels are set to **neutral** to ensure the code runs successfully.

**Step 1:** Update the path placeholder in `config.py` (replace `xxx` with your own path).

**Step 2:** Generate the processed dataset:
```bash
python toolkit/preprocess/mer2026.py
```

**Step 3:** The resulting directory structure:
```
dataset/
└── mer2026-dataset/                      # https://huggingface.co/datasets/MERChallenge/MER2026
    ├── video/                            # 132,171 samples
    ├── audio/                            # 132,171 samples
    ├── openface_face/                    # 132,171 samples
    ├── subtitle_chieng.csv               # Pre-extracted subtitles, 132,171 samples
    ├── track1_train.csv                  # Training set (same-person), 9,395 samples
    └── track1_track2_candidate.csv       # 20,000 candidate samples

## Generated processed data
└── mer2026-dataset-process/              
    ├── video/
    ├── audio/
    ├── openface_face/
    ├── track1_subtitle_chieng.csv
    └── track1_label_6way.npz
```

---


## ✨ Training & Inference

### 1. Prepare Tools

All necessary tools are available at: https://pan.baidu.com/s/13dw2qQOBxgWTr0mhEseqEQ?pwd=mn6w

Download and place them into `./tools`:

```
MER2026_Track1/
└── tools/
    ├── emonet/
    ├── ferplus/
    ├── ...
    └── transformers/
        ├── bloom-7b1/
        ├── chinese-hubert-base/
        └── ...
```

---

### 2. Feature Extraction

Use `--feature_level` with `UTTERANCE` or `FRAME` to extract utterance-level or frame-level features.

**Visual Features:**
```bash
cd feature_extraction/visual

python -u extract_emonet_embedding.py   --dataset=MER2026 --feature_level='UTTERANCE' --gpu=0
python -u extract_ferplus_embedding.py  --dataset=MER2026 --feature_level='UTTERANCE' --model='resnet50_ferplus_dag' --gpu=0
python -u extract_ferplus_embedding.py  --dataset=MER2026 --feature_level='UTTERANCE' --model='senet50_ferplus_dag'  --gpu=0
python -u extract_vision_huggingface.py --dataset=MER2026 --feature_level='UTTERANCE' --model_name='clip-vit-base-patch32'  --gpu=0
python -u extract_vision_huggingface.py --dataset=MER2026 --feature_level='UTTERANCE' --model_name='clip-vit-large-patch14' --gpu=0
python -u extract_vision_huggingface.py --dataset=MER2026 --feature_level='UTTERANCE' --model_name='videomae-base'           --gpu=0
python -u extract_vision_huggingface.py --dataset=MER2026 --feature_level='UTTERANCE' --model_name='videomae-large'          --gpu=0
```

**Audio Features:**
```bash
cd feature_extraction/audio

python -u extract_audio_huggingface.py --dataset=MER2026 --feature_level='UTTERANCE' --model_name='chinese-hubert-base'    --gpu=0
python -u extract_audio_huggingface.py --dataset=MER2026 --feature_level='UTTERANCE' --model_name='chinese-hubert-large'   --gpu=0
python -u extract_audio_huggingface.py --dataset=MER2026 --feature_level='UTTERANCE' --model_name='chinese-wav2vec2-base'  --gpu=0
python -u extract_audio_huggingface.py --dataset=MER2026 --feature_level='UTTERANCE' --model_name='chinese-wav2vec2-large' --gpu=0
python -u extract_audio_huggingface.py --dataset=MER2026 --feature_level='UTTERANCE' --model_name='wavlm-base'             --gpu=0
```

**Lexical Features:**
```bash
cd feature_extraction/text

python extract_text_huggingface.py --dataset=MER2026 --feature_level='UTTERANCE' --model_name='chinese-roberta-wwm-ext'       --gpu=0
python extract_text_huggingface.py --dataset=MER2026 --feature_level='UTTERANCE' --model_name='chinese-roberta-wwm-ext-large' --gpu=0
python extract_text_huggingface.py --dataset=MER2026 --feature_level='UTTERANCE' --model_name='chinese-macbert-base'          --gpu=0
python extract_text_huggingface.py --dataset=MER2026 --feature_level='UTTERANCE' --model_name='chinese-macbert-large'         --gpu=0
```

**Check Extraction Progress:**
```bash
python toolkit/utils/functions.py feature_extraction_progress MER2026
```

---

### 3. Unimodal Baselines

```bash
# Example: using 'chinese-hubert-base-UTT' as feature
# Hyperparameters are randomly selected during training
python -u main-release.py \
  --model='attention' \
  --feat_type='utt' \
  --dataset=MER2026 \
  --audio_feature=chinese-hubert-base-UTT \
  --text_feature=chinese-hubert-base-UTT \
  --video_feature=chinese-hubert-base-UTT \
  --gpu=0

# Run all features (50 iterations each)
bash run-unimodal.sh
```

---

### 4. Multimodal Baselines

First, update `AUDIO_RANK_LOW2HIGH`, `TEXT_RANK_LOW2HIGH`, and `IMAGE_RANK_LOW2HIGH` in `toolkit/globals.py` based on your unimodal results.

```bash
# fusion_topn: select top-N features per modality
# fusion_modality: report bimodal or trimodal results
python -u main-release.py \
  --model=attention_topn \
  --feat_type='utt' \
  --dataset=MER2026 \
  --fusion_topn=1 \
  --fusion_modality='AVT' \
  --gpu=0

# Run all features (50 iterations each)
bash run-multimodal.sh
```
