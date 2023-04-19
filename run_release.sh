########################################################################
######################## step1: dataset preprocess #####################
########################################################################
python main-baseline.py normalize_dataset_format --data_root='./dataset-release' --save_root='./dataset-process'

############################################################################
################# step2: multimodal feature extraction #####################
# you can also extract frame-level features setting --feature_level='FRAME'# 
############################################################################
## visual feature extraction
cd feature_extraction/visual
python extract_openface.py --dataset=MER2023 --type=videoOne ## run on windows => you can also utilize the linux version openFace
python -u extract_manet_embedding.py    --dataset=MER2023 --feature_level='UTTERANCE' --gpu=0
python -u extract_ferplus_embedding.py  --dataset=MER2023 --feature_level='UTTERANCE' --model_name='resnet50_ferplus_dag' --gpu=0
python -u extract_ferplus_embedding.py  --dataset=MER2023 --feature_level='UTTERANCE' --model_name='senet50_ferplus_dag'  --gpu=0
python -u extract_msceleb_embedding.py  --dataset=MER2023 --feature_level='UTTERANCE' --gpu=0
python -u extract_imagenet_embedding.py --dataset=MER2023 --feature_level='UTTERANCE' --gpu=0

## acoustic feature extraction
chmod -R 777 ./tools/ffmpeg-4.4.1-i686-static
chmod -R 777 ./tools/opensmile-2.3.0
python main-baseline.py split_audio_from_video_16k './dataset-process/video' './dataset-process/audio'
cd feature_extraction/audio
python -u extract_wav2vec_embedding.py       --dataset='MER2023' --feature_level='UTTERANCE' --gpu=0
python -u extract_transformers_embedding.py  --dataset='MER2023' --feature_level='UTTERANCE' --model_name='chinese-hubert-base'  --gpu=0
python -u extract_transformers_embedding.py  --dataset='MER2023' --feature_level='UTTERANCE' --model_name='chinese-hubert-large' --gpu=0
python -u extract_transformers_embedding.py  --dataset='MER2023' --feature_level='UTTERANCE' --model_name='chinese-wav2vec2-base'  --gpu=0
python -u extract_transformers_embedding.py  --dataset='MER2023' --feature_level='UTTERANCE' --model_name='chinese-wav2vec2-large' --gpu=0
python -u extract_vggish_embedding.py        --dataset='MER2023' --feature_level='UTTERANCE' --gpu=0
python -u handcrafted_feature_extractor.py   --dataset='MER2023' --feature_level='UTTERANCE' --feature_extractor='opensmile' --feature_set='IS09'
python -u handcrafted_feature_extractor.py   --dataset='MER2023' --feature_level='UTTERANCE' --feature_extractor='opensmile' --feature_set='IS10'
python -u handcrafted_feature_extractor.py   --dataset='MER2023' --feature_level='UTTERANCE' --feature_extractor='opensmile' --feature_set='eGeMAPS'

## lexical feature extraction
python main-baseline.py generate_transcription_files_asr   ./dataset-process/audio ./dataset-process/transcription-old.csv
python main-baseline.py refinement_transcription_files_asr ./dataset-process/transcription-old.csv ./dataset-process/transcription.csv
cd feature_extraction/text
python extract_text_embedding_LZ.py --dataset='MER2023' --feature_level='UTTERANCE' --model_name='bert-base-chinese'             --gpu=0
python extract_text_embedding_LZ.py --dataset='MER2023' --feature_level='UTTERANCE' --model_name='chinese-roberta-wwm-ext'       --gpu=0
python extract_text_embedding_LZ.py --dataset='MER2023' --feature_level='UTTERANCE' --model_name='chinese-roberta-wwm-ext-large' --gpu=0
python extract_text_embedding_LZ.py --dataset='MER2023' --feature_level='UTTERANCE' --model_name='deberta-chinese-large'         --gpu=0
python extract_text_embedding_LZ.py --dataset='MER2023' --feature_level='UTTERANCE' --model_name='chinese-electra-180g-small'    --gpu=0
python extract_text_embedding_LZ.py --dataset='MER2023' --feature_level='UTTERANCE' --model_name='chinese-electra-180g-base'     --gpu=0
python extract_text_embedding_LZ.py --dataset='MER2023' --feature_level='UTTERANCE' --model_name='chinese-electra-180g-large'    --gpu=0
python extract_text_embedding_LZ.py --dataset='MER2023' --feature_level='UTTERANCE' --model_name='chinese-xlnet-base'            --gpu=0
python extract_text_embedding_LZ.py --dataset='MER2023' --feature_level='UTTERANCE' --model_name='chinese-macbert-base'          --gpu=0
python extract_text_embedding_LZ.py --dataset='MER2023' --feature_level='UTTERANCE' --model_name='chinese-macbert-large'         --gpu=0
python extract_text_embedding_LZ.py --dataset='MER2023' --feature_level='UTTERANCE' --model_name='taiyi-clip-roberta-chinese'    --gpu=0
python extract_text_embedding_LZ.py --dataset='MER2023' --feature_level='UTTERANCE' --model_name='wenzhong2-gpt2-chinese'        --gpu=0
python extract_text_embedding_LZ.py --dataset='MER2023' --feature_level='UTTERANCE' --model_name='albert_chinese_tiny'           --gpu=0
python extract_text_embedding_LZ.py --dataset='MER2023' --feature_level='UTTERANCE' --model_name='albert_chinese_small'          --gpu=0


########################################################################
######## step3: training unimodal and multimodal classifiers ###########
########################################################################
## unimodal results: choose lr from [1e-3, 1e-4, 1e-5] and test each lr three times
python -u main-release.py --dataset='MER2023' --test_sets='test3' --audio_feature='manet_UTT' --text_feature='manet_UTT' --video_feature='manet_UTT' --lr=1e-3 --gpu=0
python -u main-release.py --dataset='MER2023' --test_sets='test3' --audio_feature='resnet50face_UTT' --text_feature='resnet50face_UTT' --video_feature='resnet50face_UTT' --lr=1e-3 --gpu=0
python -u main-release.py --dataset='MER2023' --test_sets='test3' --audio_feature='senet50face_UTT' --text_feature='senet50face_UTT' --video_feature='senet50face_UTT' --lr=1e-3 --gpu=0
python -u main-release.py --dataset='MER2023' --test_sets='test3' --audio_feature='msceleb_UTT' --text_feature='msceleb_UTT' --video_feature='msceleb_UTT' --lr=1e-3 --gpu=0
python -u main-release.py --dataset='MER2023' --test_sets='test3' --audio_feature='imagenet_UTT' --text_feature='imagenet_UTT' --video_feature='imagenet_UTT' --lr=1e-3 --gpu=0
python -u main-release.py --dataset='MER2023' --test_sets='test3' --audio_feature='bert-base-chinese-4-UTT' --text_feature='bert-base-chinese-4-UTT' --video_feature='bert-base-chinese-4-UTT' --lr=1e-3 --gpu=0
python -u main-release.py --dataset='MER2023' --test_sets='test3' --audio_feature='chinese-roberta-wwm-ext-4-UTT' --text_feature='chinese-roberta-wwm-ext-4-UTT' --video_feature='chinese-roberta-wwm-ext-4-UTT' --lr=1e-3 --gpu=0
python -u main-release.py --dataset='MER2023' --test_sets='test3' --audio_feature='chinese-roberta-wwm-ext-large-4-UTT' --text_feature='chinese-roberta-wwm-ext-large-4-UTT' --video_feature='chinese-roberta-wwm-ext-large-4-UTT' --lr=1e-3 --gpu=0
python -u main-release.py --dataset='MER2023' --test_sets='test3' --audio_feature='deberta-chinese-large-4-UTT' --text_feature='deberta-chinese-large-4-UTT' --video_feature='deberta-chinese-large-4-UTT' --lr=1e-3 --gpu=0
python -u main-release.py --dataset='MER2023' --test_sets='test3' --audio_feature='chinese-electra-180g-small-4-UTT' --text_feature='chinese-electra-180g-small-4-UTT' --video_feature='chinese-electra-180g-small-4-UTT' --lr=1e-3 --gpu=0
python -u main-release.py --dataset='MER2023' --test_sets='test3' --audio_feature='chinese-electra-180g-base-4-UTT' --text_feature='chinese-electra-180g-base-4-UTT' --video_feature='chinese-electra-180g-base-4-UTT' --lr=1e-3 --gpu=0
python -u main-release.py --dataset='MER2023' --test_sets='test3' --audio_feature='chinese-electra-180g-large-4-UTT' --text_feature='chinese-electra-180g-large-4-UTT' --video_feature='chinese-electra-180g-large-4-UTT' --lr=1e-3 --gpu=0
python -u main-release.py --dataset='MER2023' --test_sets='test3' --audio_feature='chinese-xlnet-base-4-UTT' --text_feature='chinese-xlnet-base-4-UTT' --video_feature='chinese-xlnet-base-4-UTT' --lr=1e-3 --gpu=0
python -u main-release.py --dataset='MER2023' --test_sets='test3' --audio_feature='chinese-macbert-base-4-UTT' --text_feature='chinese-macbert-base-4-UTT' --video_feature='chinese-macbert-base-4-UTT' --lr=1e-3 --gpu=0
python -u main-release.py --dataset='MER2023' --test_sets='test3' --audio_feature='chinese-macbert-large-4-UTT' --text_feature='chinese-macbert-large-4-UTT' --video_feature='chinese-macbert-large-4-UTT' --lr=1e-3 --gpu=0
python -u main-release.py --dataset='MER2023' --test_sets='test3' --audio_feature='taiyi-clip-roberta-chinese-4-UTT' --text_feature='taiyi-clip-roberta-chinese-4-UTT' --video_feature='taiyi-clip-roberta-chinese-4-UTT' --lr=1e-3 --gpu=0
python -u main-release.py --dataset='MER2023' --test_sets='test3' --audio_feature='wenzhong2-gpt2-chinese-4-UTT' --text_feature='wenzhong2-gpt2-chinese-4-UTT' --video_feature='wenzhong2-gpt2-chinese-4-UTT' --lr=1e-3 --gpu=0
python -u main-release.py --dataset='MER2023' --test_sets='test3' --audio_feature='albert_chinese_tiny-4-UTT' --text_feature='albert_chinese_tiny-4-UTT' --video_feature='albert_chinese_tiny-4-UTT' --lr=1e-3 --gpu=0
python -u main-release.py --dataset='MER2023' --test_sets='test3' --audio_feature='albert_chinese_small-4-UTT' --text_feature='albert_chinese_small-4-UTT' --video_feature='albert_chinese_small-4-UTT' --lr=1e-3 --gpu=0
python -u main-release.py --dataset='MER2023' --test_sets='test3' --audio_feature='wav2vec-large-c-UTT' --text_feature='wav2vec-large-c-UTT' --video_feature='wav2vec-large-c-UTT' --lr=1e-3 --gpu=0
python -u main-release.py --dataset='MER2023' --test_sets='test3' --audio_feature='wav2vec-large-z-UTT' --text_feature='wav2vec-large-z-UTT' --video_feature='wav2vec-large-z-UTT' --lr=1e-3 --gpu=0
python -u main-release.py --dataset='MER2023' --test_sets='test3' --audio_feature='chinese-hubert-base-UTT' --text_feature='chinese-hubert-base-UTT' --video_feature='chinese-hubert-base-UTT' --lr=1e-3 --gpu=0
python -u main-release.py --dataset='MER2023' --test_sets='test3' --audio_feature='chinese-hubert-large-UTT' --text_feature='chinese-hubert-large-UTT' --video_feature='chinese-hubert-large-UTT' --lr=1e-3 --gpu=0
python -u main-release.py --dataset='MER2023' --test_sets='test3' --audio_feature='chinese-wav2vec2-base-UTT' --text_feature='chinese-wav2vec2-base-UTT' --video_feature='chinese-wav2vec2-base-UTT' --lr=1e-3 --gpu=0
python -u main-release.py --dataset='MER2023' --test_sets='test3' --audio_feature='chinese-wav2vec2-large-UTT' --text_feature='chinese-wav2vec2-large-UTT' --video_feature='chinese-wav2vec2-large-UTT' --lr=1e-3 --gpu=0
python -u main-release.py --dataset='MER2023' --test_sets='test3' --audio_feature='vggish_UTT' --text_feature='vggish_UTT' --video_feature='vggish_UTT' --lr=1e-3 --gpu=0
python -u main-release.py --dataset='MER2023' --test_sets='test3' --audio_feature='IS09_UTT' --text_feature='IS09_UTT' --video_feature='IS09_UTT' --lr=1e-3 --gpu=0
python -u main-release.py --dataset='MER2023' --test_sets='test3' --audio_feature='IS10_UTT' --text_feature='IS10_UTT' --video_feature='IS10_UTT' --lr=1e-3 --gpu=0
python -u main-release.py --dataset='MER2023' --test_sets='test3' --audio_feature='eGeMAPS_UTT' --text_feature='eGeMAPS_UTT' --video_feature='eGeMAPS_UTT' --lr=1e-3 --gpu=0

## multimodal results: choose lr from [1e-3, 1e-4, 1e-5] and test each lr three times
python -u main-release.py --dataset='MER2023' --model_type='attention' --test_sets='test3' --audio_feature='chinese-hubert-large-UTT' --text_feature='chinese-macbert-large-4-UTT' --video_feature='chinese-macbert-large-4-UTT' --lr=1e-3 --gpu=0
python -u main-release.py --dataset='MER2023' --model_type='attention' --test_sets='test3' --audio_feature='chinese-hubert-large-UTT' --text_feature='chinese-roberta-wwm-ext-large-4-UTT' --video_feature='chinese-roberta-wwm-ext-large-4-UTT' --lr=1e-3 --gpu=0
python -u main-release.py --dataset='MER2023' --model_type='attention' --test_sets='test3' --audio_feature='chinese-hubert-large-UTT' --text_feature='chinese-macbert-base-4-UTT' --video_feature='chinese-macbert-base-4-UTT' --lr=1e-3 --gpu=0
python -u main-release.py --dataset='MER2023' --model_type='attention' --test_sets='test3' --audio_feature='chinese-hubert-large-UTT' --text_feature='resnet50face_UTT' --video_feature='resnet50face_UTT' --lr=1e-3 --gpu=0
python -u main-release.py --dataset='MER2023' --model_type='attention' --test_sets='test3' --audio_feature='chinese-hubert-large-UTT' --text_feature='manet_UTT' --video_feature='manet_UTT' --lr=1e-3 --gpu=0
python -u main-release.py --dataset='MER2023' --model_type='attention' --test_sets='test3' --audio_feature='chinese-hubert-base-UTT' --text_feature='chinese-macbert-large-4-UTT' --video_feature='chinese-macbert-large-4-UTT' --lr=1e-3 --gpu=0
python -u main-release.py --dataset='MER2023' --model_type='attention' --test_sets='test3' --audio_feature='chinese-hubert-base-UTT' --text_feature='chinese-roberta-wwm-ext-large-4-UTT' --video_feature='chinese-roberta-wwm-ext-large-4-UTT' --lr=1e-3 --gpu=0
python -u main-release.py --dataset='MER2023' --model_type='attention' --test_sets='test3' --audio_feature='chinese-hubert-base-UTT' --text_feature='chinese-macbert-base-4-UTT' --video_feature='chinese-macbert-base-4-UTT' --lr=1e-3 --gpu=0
python -u main-release.py --dataset='MER2023' --model_type='attention' --test_sets='test3' --audio_feature='chinese-hubert-base-UTT' --text_feature='resnet50face_UTT' --video_feature='resnet50face_UTT' --lr=1e-3 --gpu=0
python -u main-release.py --dataset='MER2023' --model_type='attention' --test_sets='test3' --audio_feature='chinese-hubert-base-UTT' --text_feature='manet_UTT' --video_feature='manet_UTT' --lr=1e-3 --gpu=0
python -u main-release.py --dataset='MER2023' --model_type='attention' --test_sets='test3' --audio_feature='chinese-macbert-large-4-UTT' --text_feature='resnet50face_UTT' --video_feature='resnet50face_UTT' --lr=1e-3 --gpu=0
python -u main-release.py --dataset='MER2023' --model_type='attention' --test_sets='test3' --audio_feature='chinese-macbert-large-4-UTT' --text_feature='manet_UTT' --video_feature='manet_UTT' --lr=1e-3 --gpu=0
python -u main-release.py --dataset='MER2023' --model_type='attention' --test_sets='test3' --audio_feature='chinese-roberta-wwm-ext-large-4-UTT' --text_feature='resnet50face_UTT' --video_feature='resnet50face_UTT' --lr=1e-3 --gpu=0
python -u main-release.py --dataset='MER2023' --model_type='attention' --test_sets='test3' --audio_feature='chinese-roberta-wwm-ext-large-4-UTT' --text_feature='manet_UTT' --video_feature='manet_UTT' --lr=1e-3 --gpu=0
python -u main-release.py --dataset='MER2023' --model_type='attention' --test_sets='test3' --audio_feature='chinese-macbert-base-4-UTT' --text_feature='resnet50face_UTT' --video_feature='resnet50face_UTT' --lr=1e-3 --gpu=0
python -u main-release.py --dataset='MER2023' --model_type='attention' --test_sets='test3' --audio_feature='chinese-macbert-base-4-UTT' --text_feature='manet_UTT' --video_feature='manet_UTT' --lr=1e-3 --gpu=0
python -u main-release.py --dataset='MER2023' --model_type='attention' --test_sets='test3' --audio_feature='chinese-hubert-large-UTT' --text_feature='chinese-macbert-large-4-UTT' --video_feature='resnet50face_UTT' --lr=1e-3 --gpu=0
python -u main-release.py --dataset='MER2023' --model_type='attention' --test_sets='test3' --audio_feature='chinese-hubert-large-UTT' --text_feature='chinese-macbert-large-4-UTT' --video_feature='manet_UTT' --lr=1e-3 --gpu=0
python -u main-release.py --dataset='MER2023' --model_type='attention' --test_sets='test3' --audio_feature='chinese-hubert-base-UTT' --text_feature='chinese-macbert-large-4-UTT' --video_feature='resnet50face_UTT' --lr=1e-3 --gpu=0
python -u main-release.py --dataset='MER2023' --model_type='attention' --test_sets='test3' --audio_feature='chinese-hubert-large-UTT' --text_feature='chinese-roberta-wwm-ext-large-4-UTT' --video_feature='resnet50face_UTT' --lr=1e-3 --gpu=0
python -u main-release.py --dataset='MER2023' --model_type='attention' --test_sets='test3' --audio_feature='chinese-hubert-large-UTT' --text_feature='chinese-macbert-base-4-UTT' --video_feature='resnet50face_UTT' --lr=1e-3 --gpu=0


###########################
######## others ###########
###########################
## data corruption methods: corrupt videos in video_root, and save to save_root
python main-corrupt.py main_mixture_multiprocess(video_root, save_root)

## submission format
step1: "write_to_csv_pred(name2preds, pred_path)" in main-release.py
step2: submit "pred_path"

## evaluation metrics
for [test1, test2] => "report_results_on_test1_test2(label_path, pred_path)" in main-release.py 
for [test3]        => "report_results_on_test3(label_path, pred_path)"       in main-release.py 
