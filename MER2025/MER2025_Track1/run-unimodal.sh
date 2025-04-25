#!/bin/bash

names=('bloom-7b1-UTT'
       'chinese-macbert-large-UTT'
       'chinese-wav2vec2-large-UTT' 
       'emonet_UTT' 
       'videomae-base-UTT'
       'chinese-hubert-base-UTT' 
       'chinese-roberta-wwm-ext-large-UTT'
       'clip-vit-base-patch32-UTT' 
       'manet_UTT'
       'videomae-large-UTT' 
       'chinese-hubert-large-UTT' 
       'chinese-roberta-wwm-ext-UTT' 
       'clip-vit-large-patch14-UTT' 
       'resnet50face_UTT' 
       'wavlm-base-UTT' 
       'chinese-macbert-base-UTT' 
       'chinese-wav2vec2-base-UTT'
       'dinov2-large-UTT' 
       'senet50face_UTT' 
       'whisper-large-v2-UTT')


for ((i=1; i<=50; i++)); do
    echo "=== Loop index: $i/50 ==="
    
    for n in "${names[@]}"; do
        echo "feature name: $n"
        python -u main-release.py --model='attention' --feat_type='utt' --dataset=MER2025 --audio_feature="$n" --text_feature="$n" --video_feature="$n" --gpu=0
    done
    
done

echo "Finish！"