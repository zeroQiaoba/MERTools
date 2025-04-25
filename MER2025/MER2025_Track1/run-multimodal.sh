#!/bin/bash

names=('AVT' 'AV' 'AT' 'VT')
topn=(1 2)

for ((i=1; i<=50; i++)); do
    echo "=== 第 $i 次循环 ==="

    for name in "${names[@]}"; do
        echo "Fusion type: $name"

        for top in "${topn[@]}"; do
            echo "topn: $top"
            python -u main-release.py --model=attention_topn --feat_type='utt' --dataset=MER2025 --fusion_topn=$top --fusion_modality=$name --gpu=1
        done

    done
    
done

echo "Finish！"