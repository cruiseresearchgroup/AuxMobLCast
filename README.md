# AuxMobLCast

The code for Paper Leveraging Language Foundation Models for Human Mobility Forecasting (ACM SigSpatial 2022)


data: Contains the example prompt files. At the current stage, we only provide one example in the folder. Please use the proposed template in our paper to process your data. (Please register for SafeGraph for more data :blush:)

logs: training and validation predictions are stored here

logger: To store the logging file (train and eval loss, accuracy, rmse, mae etc)

encoders: Contains code for XLNet and Roberta encoders

bert_encoder.py: Code file for bert encoder (Inspired from huggingface bert file with slight modifications)

dataloader.py: Code to load the data from text files

dataset.py: Code to create a dataset from text files (involves tokenization and padding)

encoder_decoder.py: This python code is inspired from huggingface with slight modifications. It is used to build the encoder-decoder model

finetune.py: Contains training code

finetune_test.py: Contains inference code

metrics: Contains code for metrics such as RMSE and MAE

poi_dict.json: Json file with keys as the POI (NYC as example) and value as the ID. Used for POI classification.
