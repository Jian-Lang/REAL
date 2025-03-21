#! /bin/bash

python preprocess/generate_caption_BLIP.py  # generate caption for each video
python preprocess/generate_query_text.py    # generate query text for each video
python preprocess/make_retrieval_tensor.py  # make retrieval tensor for each video

python retrieve/conduct_retrieval.py        # conduct retrieval for each video