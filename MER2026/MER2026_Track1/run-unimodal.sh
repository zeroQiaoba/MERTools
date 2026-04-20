#!/bin/bash

## finish feature extraction
names=('clip-vit-base-patch32-UTT'
       'clip-vit-large-patch14-UTT'
       'videomae-base-UTT'
       'videomae-large-UTT'
       'chinese-hubert-base-UTT'
       'wavlm-base-UTT'
       'chinese-hubert-large-UTT'
       'chinese-wav2vec2-large-UTT'
       'chinese-wav2vec2-base-UTT'
       'chinese-roberta-wwm-ext-UTT'
       'chinese-roberta-wwm-ext-large-UTT'
       'chinese-macbert-base-UTT'
       'chinese-macbert-large-UTT'
       'emonet_UTT'
       'resnet50face_UTT'
       'senet50face_UTT')

## unfinish feature extraction
# 'clip-vit-large-patch14-UTT' 1790

for ((i=1; i<=50; i++)); do
    echo "=== Loop index: $i/50 ==="
    
    for n in "${names[@]}"; do
        echo "feature name: $n"
        python -u main-release.py --model='attention' --feat_type='utt' --dataset=MER2026 --audio_feature="$n" --text_feature="$n" --video_feature="$n" --gpu=0
    done
    
done

echo "Finish！"