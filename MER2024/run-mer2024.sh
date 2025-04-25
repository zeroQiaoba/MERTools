#############################################################################################
## Baseline for MER2024
#############################################################################################
step0: download the dataset and put it into './mer2024-dataset'


step1: dataset preprocess 
# see toolkit/preprocess/mer2024.py


step2: visual process 
2.1 face extraction
# cd feature_extraction/visual
# python extract_openface.py --dataset=MER2024 --type=videoOne
# mv ./mer2024-dataset-process/features/openface_face ./mer2024-dataset-process

2.2 visual feature extraction
# cd feature_extraction/visual
# python -u extract_manet_embedding.py    --dataset=MER2024 --feature_level='UTTERANCE'                                                 --gpu=0                                                
# python -u extract_emonet_embedding.py   --dataset=MER2024 --feature_level='UTTERANCE'                                                 --gpu=0                                                
# python -u extract_ferplus_embedding.py  --dataset=MER2024 --feature_level='UTTERANCE' --model='resnet50_ferplus_dag'                  --gpu=0                  
# python -u extract_ferplus_embedding.py  --dataset=MER2024 --feature_level='UTTERANCE' --model='senet50_ferplus_dag'                   --gpu=0                 
# python -u extract_vision_huggingface.py --dataset=MER2024 --feature_level='UTTERANCE' --model_name='clip-vit-base-patch32'            --gpu=0            
# python -u extract_vision_huggingface.py --dataset=MER2024 --feature_level='UTTERANCE' --model_name='clip-vit-large-patch14'           --gpu=0           
# python -u extract_vision_huggingface.py --dataset=MER2024 --feature_level='UTTERANCE' --model_name='eva02_base_patch14_224.mim_in22k' --gpu=0 
# python -u extract_vision_huggingface.py --dataset=MER2024 --feature_level='UTTERANCE' --model_name='videomae-base'                    --gpu=0                   
# python -u extract_vision_huggingface.py --dataset=MER2024 --feature_level='UTTERANCE' --model_name='videomae-large'                   --gpu=0                   
# python -u extract_vision_huggingface.py --dataset=MER2024 --feature_level='UTTERANCE' --model_name='dinov2-large'                     --gpu=0                    
# CUDA_VISIBLE_DEVICES=0 python -u extract_sun_videomae.py --dataset MER2024 --feature_level UTTERANCE --batch_size 64 --model vit_base_patch16_160 --input_size 160 --short_side_size 160 --finetune tools/videomae-base-VoxCeleb2/checkpoint-99.pth
# CUDA_VISIBLE_DEVICES=0 python -u extract_sun_videomae.py --dataset MER2024 --feature_level UTTERANCE --batch_size 64 --model vit_base_patch16_224 --input_size 224 --short_side_size 224 --finetune tools/videomae-base-K400-mer2023/checkpoint-299.pth


step3: audio process
3.1 audio extraction
# python toolkit/utils/functions.py func_split_audio_from_video_16k './mer2024-dataset-process/video' './mer2024-dataset-process/audio'

3.2 audio feature extraction
# cd feature_extraction/audio
# python -u handcrafted_feature_extractor.py       --dataset='MER2024' --feature_level='UTTERANCE' --feature_extractor='opensmile' --feature_set='eGeMAPS' 
# python -u extract_vggish_embedding.py            --dataset='MER2024' --feature_level='UTTERANCE'                                         --gpu=0
# python -u extract_audio_huggingface.py           --dataset='MER2024' --feature_level='UTTERANCE' --model_name='chinese-hubert-base'      --gpu=0
# python -u extract_audio_huggingface.py           --dataset='MER2024' --feature_level='UTTERANCE' --model_name='chinese-hubert-large'     --gpu=0
# python -u extract_audio_huggingface.py           --dataset='MER2024' --feature_level='UTTERANCE' --model_name='chinese-wav2vec2-base'    --gpu=0
# python -u extract_audio_huggingface.py           --dataset='MER2024' --feature_level='UTTERANCE' --model_name='chinese-wav2vec2-large'   --gpu=0
# python -u extract_audio_huggingface.py           --dataset='MER2024' --feature_level='UTTERANCE' --model_name='whisper-base'             --gpu=0
# python -u extract_audio_huggingface.py           --dataset='MER2024' --feature_level='UTTERANCE' --model_name='whisper-large-v2'         --gpu=0
# python -u extract_emotion2vec.py                 --dataset='MER2024' --feature_level='UTTERANCE'                                        


step4: text process
4.1 subtitle extraction
# python main-asr.py generate_transcription_files_asr   ./mer2024-dataset-process/audio ./mer2024-dataset-process/transcription-old.csv # audio -> subtitle
# python main-asr.py refinement_transcription_files_asr ./mer2024-dataset-process/transcription-old.csv ./mer2024-dataset-process/transcription.csv # subtitle + punctuation
# python main-asr.py merge_trans_with_checked ./mer2024-dataset-process/transcription.csv ./mer2024-dataset/label-transcription.csv ./mer2024-dataset-process/transcription-merge.csv # use checked subtile to correct asr outputs

4.2 lexical feature extraction
# cd feature_extraction/text
# python extract_text_huggingface.py --dataset='MER2024' --feature_level='UTTERANCE' --model_name='chinese-roberta-wwm-ext'               --gpu=0
# python extract_text_huggingface.py --dataset='MER2024' --feature_level='UTTERANCE' --model_name='chinese-roberta-wwm-ext-large'         --gpu=0
# python extract_text_huggingface.py --dataset='MER2024' --feature_level='UTTERANCE' --model_name='chinese-electra-180g-base'             --gpu=0
# python extract_text_huggingface.py --dataset='MER2024' --feature_level='UTTERANCE' --model_name='chinese-electra-180g-large'            --gpu=0
# python extract_text_huggingface.py --dataset='MER2024' --feature_level='UTTERANCE' --model_name='chinese-xlnet-base'                    --gpu=0
# python extract_text_huggingface.py --dataset='MER2024' --feature_level='UTTERANCE' --model_name='chinese-macbert-base'                  --gpu=0
# python extract_text_huggingface.py --dataset='MER2024' --feature_level='UTTERANCE' --model_name='chinese-macbert-large'                 --gpu=0
# python extract_text_huggingface.py --dataset='MER2024' --feature_level='UTTERANCE' --model_name='chinese-pert-base'                     --gpu=0
# python extract_text_huggingface.py --dataset='MER2024' --feature_level='UTTERANCE' --model_name='chinese-pert-large'                    --gpu=0
# python extract_text_huggingface.py --dataset='MER2024' --feature_level='UTTERANCE' --model_name='chinese-lert-base'                     --gpu=0
# python extract_text_huggingface.py --dataset='MER2024' --feature_level='UTTERANCE' --model_name='chinese-lert-large'                    --gpu=0
# python extract_text_huggingface.py --dataset='MER2024' --feature_level='UTTERANCE' --model_name='albert_chinese_small'                  --gpu=0
# python extract_text_huggingface.py --dataset='MER2024' --feature_level='UTTERANCE' --model_name='paraphrase-multilingual-mpnet-base-v2' --gpu=0
# python extract_text_huggingface.py --dataset='MER2024' --feature_level='UTTERANCE' --model_name='moss-base-7b'                          --gpu=0  
# python extract_text_huggingface.py --dataset='MER2024' --feature_level='UTTERANCE' --model_name='bloom-7b1'                             --gpu=0  
# python extract_text_huggingface.py --dataset='MER2024' --feature_level='UTTERANCE' --model_name='chatglm2-6b'                           --gpu=0  
# python extract_text_huggingface.py --dataset='MER2024' --feature_level='UTTERANCE' --model_name='Baichuan-13B-Base'                     --gpu=0


step5: unimodal baseline => test different unimodal features
# python -u main-release.py --model='attention' --feat_type='utt' --dataset=MER2024 --audio_feature=eGeMAPS_UTT --text_feature=eGeMAPS_UTT --video_feature=eGeMAPS_UTT --gpu=0


step6: multimodal baseline => [fusion_topn 1~6; fusion_modality in ['AV', 'AT', 'VT', 'AVT']]
6.1 adjust AUDIO_RANK_LOW2HIGH  / TEXT_RANK_LOW2HIGH / IMAGR_RANK_LOW2HIGH  in toolkit/globals.py

6.2 training => automatic selection top-n features for each modality
# python -u main-release.py --model=attention_topn --feat_type='utt' --dataset=MER2024 --fusion_topn=1 --fusion_modality='AVT' --gpu=0
# python -u main-release.py --model=attention_topn --feat_type='utt' --dataset=MER2024 --fusion_topn=2 --fusion_modality='AVT' --gpu=0
# python -u main-release.py --model=attention_topn --feat_type='utt' --dataset=MER2024 --fusion_topn=3 --fusion_modality='AVT' --gpu=0
# python -u main-release.py --model=attention_topn --feat_type='utt' --dataset=MER2024 --fusion_topn=4 --fusion_modality='AVT' --gpu=0
# python -u main-release.py --model=attention_topn --feat_type='utt' --dataset=MER2024 --fusion_topn=5 --fusion_modality='AVT' --gpu=0
# python -u main-release.py --model=attention_topn --feat_type='utt' --dataset=MER2024 --fusion_topn=6 --fusion_modality='AVT' --gpu=0

# python -u main-release.py --model=attention_topn --feat_type='utt' --dataset=MER2024 --fusion_topn=6 --fusion_modality='AV' --gpu=0
# python -u main-release.py --model=attention_topn --feat_type='utt' --dataset=MER2024 --fusion_topn=6 --fusion_modality='AT' --gpu=0
# python -u main-release.py --model=attention_topn --feat_type='utt' --dataset=MER2024 --fusion_topn=6 --fusion_modality='VT' --gpu=0