######################################################################################################################
############################ 8 datasets + multiple features + multiple fusion strategies #############################
######################################################################################################################
step1: dataset preprocess [see toolkit/process]
# see toolkit/proprocess/mer2023.py 
# see toolkit/proprocess/sims.py
# see toolkit/proprocess/simsv2.py
# see toolkit/proprocess/cmumosi.py
# see toolkit/proprocess/cmumosei.py
# see toolkit/proprocess/meld.py
# see toolkit/proprocess/iemocap.py

step2: visual feature process
2.1 extract face
# cd feature_extraction/visual
# python extract_openface.py --dataset=MER2023 --type=videoOne ## run on windows

2.2 extract features
# cd feature_extraction/visual
# python -u extract_manet_embedding.py    --dataset=MER2023 --feature_level='UTTERANCE'                                                 --gpu=0                                                
# python -u extract_emonet_embedding.py   --dataset=MER2023 --feature_level='UTTERANCE'                                                 --gpu=0                                                
# python -u extract_ferplus_embedding.py  --dataset=MER2023 --feature_level='UTTERANCE' --model='resnet50_ferplus_dag'                  --gpu=0                 
# python -u extract_ferplus_embedding.py  --dataset=MER2023 --feature_level='UTTERANCE' --model='senet50_ferplus_dag'                   --gpu=0                 
# python -u extract_msceleb_embedding.py  --dataset=MER2023 --feature_level='UTTERANCE'                                                 --gpu=0                                                
# python -u extract_imagenet_embedding.py --dataset=MER2023 --feature_level='UTTERANCE'                                                 --gpu=0                                                
# python -u extract_vision_huggingface.py --dataset=MER2023 --feature_level='UTTERANCE' --model_name='clip-vit-base-patch32'            --gpu=0           
# python -u extract_vision_huggingface.py --dataset=MER2023 --feature_level='UTTERANCE' --model_name='clip-vit-large-patch14'           --gpu=0          
# python -u extract_vision_huggingface.py --dataset=MER2023 --feature_level='UTTERANCE' --model_name='eva02_base_patch14_224.mim_in22k' --gpu=0
# python -u extract_vision_huggingface.py --dataset=MER2023 --feature_level='UTTERANCE' --model_name='data2vec-vision-base-ft1k'        --gpu=0
# python -u extract_vision_huggingface.py --dataset=MER2023 --feature_level='UTTERANCE' --model_name='videomae-base'                    --gpu=0                  
# python -u extract_vision_huggingface.py --dataset=MER2023 --feature_level='UTTERANCE' --model_name='videomae-large'                   --gpu=0                  
# python -u extract_vision_huggingface.py --dataset=MER2023 --feature_level='UTTERANCE' --model_name='dinov2-large'                     --gpu=0               
# python -u extract_vision_huggingface.py --dataset=MER2023 --feature_level='UTTERANCE' --model_name='dinov2-giant'                     --gpu=0          

step3: lexical feature process
# cd feature_extraction/text
# python extract_text_huggingface.py --dataset='MER2023' --feature_level='UTTERANCE' --model_name='bert-base-chinese'                     --gpu=0
# python extract_text_huggingface.py --dataset='MER2023' --feature_level='UTTERANCE' --model_name='chinese-roberta-wwm-ext'               --gpu=0
# python extract_text_huggingface.py --dataset='MER2023' --feature_level='UTTERANCE' --model_name='chinese-roberta-wwm-ext-large'         --gpu=0
# python extract_text_huggingface.py --dataset='MER2023' --feature_level='UTTERANCE' --model_name='deberta-chinese-large'                 --gpu=0
# python extract_text_huggingface.py --dataset='MER2023' --feature_level='UTTERANCE' --model_name='chinese-electra-180g-small'            --gpu=0
# python extract_text_huggingface.py --dataset='MER2023' --feature_level='UTTERANCE' --model_name='chinese-electra-180g-base'             --gpu=0
# python extract_text_huggingface.py --dataset='MER2023' --feature_level='UTTERANCE' --model_name='chinese-electra-180g-large'            --gpu=0
# python extract_text_huggingface.py --dataset='MER2023' --feature_level='UTTERANCE' --model_name='chinese-xlnet-base'                    --gpu=0
# python extract_text_huggingface.py --dataset='MER2023' --feature_level='UTTERANCE' --model_name='chinese-macbert-base'                  --gpu=0
# python extract_text_huggingface.py --dataset='MER2023' --feature_level='UTTERANCE' --model_name='chinese-macbert-large'                 --gpu=0
# python extract_text_huggingface.py --dataset='MER2023' --feature_level='UTTERANCE' --model_name='chinese-pert-base'                     --gpu=0
# python extract_text_huggingface.py --dataset='MER2023' --feature_level='UTTERANCE' --model_name='chinese-pert-large'                    --gpu=0
# python extract_text_huggingface.py --dataset='MER2023' --feature_level='UTTERANCE' --model_name='chinese-lert-small'                    --gpu=0
# python extract_text_huggingface.py --dataset='MER2023' --feature_level='UTTERANCE' --model_name='chinese-lert-base'                     --gpu=0
# python extract_text_huggingface.py --dataset='MER2023' --feature_level='UTTERANCE' --model_name='chinese-lert-large'                    --gpu=0
# python extract_text_huggingface.py --dataset='MER2023' --feature_level='UTTERANCE' --model_name='gpt2-chinese-cluecorpussmall'          --gpu=0
# python extract_text_huggingface.py --dataset='MER2023' --feature_level='UTTERANCE' --model_name='taiyi-clip-roberta-chinese'            --gpu=0
# python extract_text_huggingface.py --dataset='MER2023' --feature_level='UTTERANCE' --model_name='wenzhong2-gpt2-chinese'                --gpu=0
# python extract_text_huggingface.py --dataset='MER2023' --feature_level='UTTERANCE' --model_name='albert_chinese_tiny'                   --gpu=0
# python extract_text_huggingface.py --dataset='MER2023' --feature_level='UTTERANCE' --model_name='albert_chinese_small'                  --gpu=0
# python extract_text_huggingface.py --dataset='MER2023' --feature_level='UTTERANCE' --model_name='simbert-base-chinese'                  --gpu=0
# python extract_text_huggingface.py --dataset='MER2023' --feature_level='UTTERANCE' --model_name='paraphrase-multilingual-mpnet-base-v2' --gpu=0
# python extract_text_huggingface.py --dataset='MER2023' --feature_level='UTTERANCE' --model_name='llama-7b-hf'                           --gpu=0  
# python extract_text_huggingface.py --dataset='MER2023' --feature_level='UTTERANCE' --model_name='llama-13b-hf'                          --gpu=0  
# python extract_text_huggingface.py --dataset='MER2023' --feature_level='UTTERANCE' --model_name='llama-2-7b'                            --gpu=0  
# python extract_text_huggingface.py --dataset='MER2023' --feature_level='UTTERANCE' --model_name='Llama-2-13b-hf'                        --gpu=0  
# python extract_text_huggingface.py --dataset='MER2023' --feature_level='UTTERANCE' --model_name='chinese-alpaca-2-13b'                  --gpu=0  
# python extract_text_huggingface.py --dataset='MER2023' --feature_level='UTTERANCE' --model_name='moss-base-7b'                          --gpu=0  
# python extract_text_huggingface.py --dataset='MER2023' --feature_level='UTTERANCE' --model_name='stablelm-base-alpha-7b-v2'             --gpu=0  
# python extract_text_huggingface.py --dataset='MER2023' --feature_level='UTTERANCE' --model_name='bloom-7b1'                             --gpu=0  
# python extract_text_huggingface.py --dataset='MER2023' --feature_level='UTTERANCE' --model_name='vicuna-7b-v0'                          --gpu=0  
# python extract_text_huggingface.py --dataset='MER2023' --feature_level='UTTERANCE' --model_name='stable-vicuna-13b'                     --gpu=0  
# python extract_text_huggingface.py --dataset='MER2023' --feature_level='UTTERANCE' --model_name='chatglm2-6b'                           --gpu=0  
# python extract_text_huggingface.py --dataset='MER2023' --feature_level='UTTERANCE' --model_name='Baichuan-7B'                           --gpu=0  
# python extract_text_huggingface.py --dataset='MER2023' --feature_level='UTTERANCE' --model_name='Baichuan-13B-Base'                     --gpu=0  
# python extract_text_huggingface.py --dataset='MER2023' --feature_level='UTTERANCE' --model_name='opt-13b'                               --gpu=0  
# python extract_text_huggingface.py --dataset='MER2023' --feature_level='UTTERANCE' --model_name='Baichuan2-7B-Base'                     --gpu=0
# python extract_text_huggingface.py --dataset='MER2023' --feature_level='UTTERANCE' --model_name='Baichuan2-13B-Base'                    --gpu=0  
# python extract_text_huggingface.py --dataset='MER2023' --feature_level='UTTERANCE' --model_name='falcon-7b'                             --gpu=0 

step4: audio feature process
4.1 split into 16kHz audio
# python toolkit/utils/functions.py func_split_audio_from_video_16k 'dataset/mer2023-dataset-process/video' 'dataset/mer2023-dataset-process/audio'
# python toolkit/utils/functions.py func_split_audio_from_video_16k 'dataset/sims-process/video' 'dataset/sims-process/audio'
# python toolkit/utils/functions.py func_split_audio_from_video_16k 'dataset/meld-process/subvideo' 'dataset/meld-process/subaudio'
# python toolkit/utils/functions.py func_split_audio_from_video_16k 'dataset/simsv2-process/video' 'dataset/simsv2-process/audio'
4.2 extract features
# cd feature_extraction/audio
# python -u extract_wav2vec_embedding.py     --dataset='MER2023' --feature_level='UTTERANCE'                                         --gpu=0
# python -u extract_vggish_embedding.py      --dataset='MER2023' --feature_level='UTTERANCE'                                         --gpu=0
# python -u extract_audio_huggingface.py     --dataset='MER2023' --feature_level='UTTERANCE' --model_name='chinese-hubert-base'      --gpu=0
# python -u extract_audio_huggingface.py     --dataset='MER2023' --feature_level='UTTERANCE' --model_name='chinese-hubert-large'     --gpu=0
# python -u extract_audio_huggingface.py     --dataset='MER2023' --feature_level='UTTERANCE' --model_name='chinese-wav2vec2-base'    --gpu=0
# python -u extract_audio_huggingface.py     --dataset='MER2023' --feature_level='UTTERANCE' --model_name='chinese-wav2vec2-large'   --gpu=0
# python -u extract_audio_huggingface.py     --dataset='MER2023' --feature_level='UTTERANCE' --model_name='wav2vec2-base-960h'       --gpu=0
# python -u extract_audio_huggingface.py     --dataset='MER2023' --feature_level='UTTERANCE' --model_name='wav2vec2-large-960h'      --gpu=0
# python -u extract_audio_huggingface.py     --dataset='MER2023' --feature_level='UTTERANCE' --model_name='whisper-base'             --gpu=0
# python -u extract_audio_huggingface.py     --dataset='MER2023' --feature_level='UTTERANCE' --model_name='whisper-large-v2'         --gpu=0
# python -u extract_audio_huggingface.py     --dataset='MER2023' --feature_level='UTTERANCE' --model_name='wavlm-base'               --gpu=0
# python -u extract_audio_huggingface.py     --dataset='MER2023' --feature_level='UTTERANCE' --model_name='wavlm-large'              --gpu=0
# python -u extract_audio_huggingface.py     --dataset='MER2023' --feature_level='UTTERANCE' --model_name='data2vec-audio-base-960h' --gpu=0
# python -u extract_audio_huggingface.py     --dataset='MER2023' --feature_level='UTTERANCE' --model_name='data2vec-audio-large'     --gpu=0
# python -u handcrafted_feature_extractor.py --dataset='MER2023' --feature_level='UTTERANCE' --feature_extractor='opensmile' --feature_set='IS09'   
# python -u handcrafted_feature_extractor.py --dataset='MER2023' --feature_level='UTTERANCE' --feature_extractor='opensmile' --feature_set='IS10'  
# python -u handcrafted_feature_extractor.py --dataset='MER2023' --feature_level='UTTERANCE' --feature_extractor='opensmile' --feature_set='IS13'    
# python -u handcrafted_feature_extractor.py --dataset='MER2023' --feature_level='UTTERANCE' --feature_extractor='opensmile' --feature_set='eGeMAPS' 


step5: unimodal benchmark
# python -u main-release.py --model='attention' --feat_type='utt' --dataset='MER2023' --audio_feature='chinese-hubert-large-UTT' --text_feature='chinese-hubert-large-UTT' --video_feature='chinese-hubert-large-UTT' --gpu=0
# python -u main-release.py --model='attention' --feat_type='utt' --dataset='MER2023' --audio_feature='clip-vit-large-patch14-UTT' --text_feature='clip-vit-large-patch14-UTT' --video_feature='clip-vit-large-patch14-UTT' --gpu=0
# python -u main-release.py --model='attention' --feat_type='utt' --dataset='MER2023' --audio_feature='Baichuan-13B-Base-UTT' --text_feature='Baichuan-13B-Base-UTT' --video_feature='Baichuan-13B-Base-UTT' --gpu=0


step6: multimodal benchmark
## for utt-level fusion
# python -u main-release.py --model='attention'   --feat_type='utt'         --dataset='MER2023' --audio_feature='chinese-hubert-large-UTT' --text_feature='Baichuan-13B-Base-UTT' --video_feature='clip-vit-large-patch14-UTT' --gpu=0
# python -u main-release.py --model='lmf'         --feat_type='utt'         --dataset='MER2023' --audio_feature='chinese-hubert-large-UTT' --text_feature='Baichuan-13B-Base-UTT' --video_feature='clip-vit-large-patch14-UTT' --gpu=0
# python -u main-release.py --model='misa'        --feat_type='utt'         --dataset='MER2023' --audio_feature='chinese-hubert-large-UTT' --text_feature='Baichuan-13B-Base-UTT' --video_feature='clip-vit-large-patch14-UTT' --gpu=0
# python -u main-release.py --model='mmim'        --feat_type='utt'         --dataset='MER2023' --audio_feature='chinese-hubert-large-UTT' --text_feature='Baichuan-13B-Base-UTT' --video_feature='clip-vit-large-patch14-UTT' --gpu=0
# python -u main-release.py --model='tfn'         --feat_type='utt'         --dataset='MER2023' --audio_feature='chinese-hubert-large-UTT' --text_feature='Baichuan-13B-Base-UTT' --video_feature='clip-vit-large-patch14-UTT' --gpu=0

## for frm_align fusion
# python -u main-release.py --model='mult'        --feat_type='frm_align'   --dataset='MER2023' --audio_feature='chinese-hubert-large-FRA' --text_feature='Baichuan-13B-Base-FRA' --video_feature='clip-vit-large-patch14-FRA' --gpu=0
# python -u main-release.py --model='mfn'         --feat_type='frm_align'   --dataset='MER2023' --audio_feature='chinese-hubert-large-FRA' --text_feature='Baichuan-13B-Base-FRA' --video_feature='clip-vit-large-patch14-FRA' --gpu=0
# python -u main-release.py --model='graph_mfn'   --feat_type='frm_align'   --dataset='MER2023' --audio_feature='chinese-hubert-large-FRA' --text_feature='Baichuan-13B-Base-FRA' --video_feature='clip-vit-large-patch14-FRA' --gpu=0
# python -u main-release.py --model='mfm'         --feat_type='frm_align'   --dataset='MER2023' --audio_feature='chinese-hubert-large-FRA' --text_feature='Baichuan-13B-Base-FRA' --video_feature='clip-vit-large-patch14-FRA' --gpu=0
# python -u main-release.py --model='mctn'        --feat_type='frm_align'   --dataset='MER2023' --audio_feature='chinese-hubert-large-FRA' --text_feature='Baichuan-13B-Base-FRA' --video_feature='clip-vit-large-patch14-FRA' --gpu=0


step7: cross-corpus benchmark
## test for sentiment strength, we take SIMS -> CMUMOSI for example
# python -u main-release.py --model=attention --feat_type='utt' --train_dataset='SIMS' --test_dataset='CMUMOSI'  --audio_feature=Baichuan-13B-Base-UTT    --text_feature=Baichuan-13B-Base-UTT --video_feature=Baichuan-13B-Base-UTT      --gpu=0
# python -u main-release.py --model=attention --feat_type='utt' --train_dataset='SIMS' --test_dataset='CMUMOSI'  --audio_feature=chinese-hubert-large-UTT --text_feature=Baichuan-13B-Base-UTT --video_feature=clip-vit-large-patch14-UTT --gpu=0

## test for discrete labels, we take MER2023 -> MELD for example
# python -u main-release.py --model=attention --feat_type='utt' --train_dataset='MER2023' --test_dataset='MELD'  --audio_feature=Baichuan-13B-Base-UTT    --text_feature=Baichuan-13B-Base-UTT --video_feature=Baichuan-13B-Base-UTT      --gpu=0
# python -u main-release.py --model=attention --feat_type='utt' --train_dataset='MER2023' --test_dataset='MELD'  --audio_feature=chinese-hubert-large-UTT --text_feature=Baichuan-13B-Base-UTT --video_feature=clip-vit-large-patch14-UTT --gpu=0
