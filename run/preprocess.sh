#! /bin/bash

python src/model/SVFEND/preprocess/make_vgg19_feature.py     # make vgg19 feature for each video
python src/model/SVFEND/preprocess/make_vgg_feature.py       # make vgg feature for each video
python src/model/SVFEND/preprocess/make_c3d_feature.py      # make c3d feature for each video
python src/model/SVFEND/preprocess/make_bert_feature.py      # make bert feature for each video
python src/model/SVFEND/preprocess/torchvggish/extract_vggish_pre.py  # make vggish feature for each video