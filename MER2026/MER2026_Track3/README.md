<div align="center">

# Track3 — MER-Prefer

### Baselines for Emotion Preference Prediction

</div>

---

## Table of Contents

- [Task Description](#-task-description)
- [Dataset](#-dataset)
- [Evaluation Metric](#-evaluation-metric)
- [Zero-shot Baselines](#️-zero-shot-baselines)

---

## 📌 Task Description

MER-Prefer is a **newly introduced track** based on the EmoPrefer concept. Given a **video** and **two candidate emotion descriptions** (d₁ and d₂), the model must determine **which description is preferred by human annotators**. This task plays a crucial role in training reward models capable of genuinely understanding human emotions — a key component for RLHF-style alignment in affective computing.

```
Input:  video  +  description d₁  +  description d₂
Output: which of d₁ or d₂ is preferred by humans
```

<div align="center">
  <img src="../icon/track3.png" alt="MER-Prefer Task" width="85%">
  <p><em>MER-Prefer: Given a video, the model needs to determine which emotion description is preferred by humans.</em></p>
</div>

---

## 🚀 Dataset

We provide two labeled training datasets with different annotation strategies:

| Dataset | Annotation Method | Samples |
|---|---|---|
| `track3_emoprefer.csv` (EmoPrefer-Data) | Major vote from 3 annotators | 574 |
| `track3_emopreferv2.csv` (EmoPrefer-Data-V2) | Single annotator | 2,096 |

```
dataset/
└── mer2026-dataset/                      # https://huggingface.co/datasets/MERChallenge/MER2026
    ├── video/                            # 132,171 samples
    ├── audio/                            # 132,171 samples
    ├── openface_face/                    # 132,171 samples
    ├── subtitle_chieng.csv               # Pre-extracted subtitles, 132,171 samples
    ├── track3_emoprefer.csv              # EmoPrefer-Data, major vote, 574 samples
    ├── track3_emopreferv2.csv            # EmoPrefer-Data-V2, single annotator, 2,096 samples
    └── track3_candidate.csv              # 10,000 pairs
```

---

## 📏 Evaluation Metric

This is a **2-class classification** task (d₁ preferred vs. d₂ preferred):

- **Primary metric:** Weighted Average F1-score (**WAF**)
- **Secondary metric:** Accuracy (**ACC**)

These metrics quantify the consistency between model predictions and human preferences.

---


## 🗝️ Zero-shot Baselines

### Preparation

```
MER2026_Track3/
└── models/
    ├── Qwen2.5-Omni-7B/
    ├── Qwen2.5-7B-Instruct/
    ├── Qwen2.5-VL-7B-Instruct/
    ├── Qwen2-Audio-7B-Instruct/
    ├── LLaVA-NeXT-Video-7B-hf/
    ├── Video-LLaVA-7B/
    └── llama-vid-7b-full-224-video-fps-1/
```

### Inference

```bash
conda activate vllm3
CUDA_VISIBLE_DEVICES=0 python -u main_dpo_batch.py  --model='qwen25vl_7b'   --output_type='MER2026Track3'
CUDA_VISIBLE_DEVICES=0 python -u main_dpo_batch.py  --model='qwen25omni_7b' --input_type='audiovideo' --output_type='MER2026Track3'
CUDA_VISIBLE_DEVICES=0 python -u main_dpo_sample.py --model='qwen2audio'        --output_type='MER2026Track3'
CUDA_VISIBLE_DEVICES=0 python -u main_dpo_sample.py --model='llavanextvideo_7b' --output_type='MER2026Track3'


conda activate whisperx
CUDA_VISIBLE_DEVICES=0 python -u main_dpo_sample.py --model='videollava' --output_type='MER2026Track3'
CUDA_VISIBLE_DEVICES=0 python -u main_dpo_sample.py --model='llamavid'   --output_type='MER2026Track3'
```

### Score Calculation

```bash
python statistic.py
```
