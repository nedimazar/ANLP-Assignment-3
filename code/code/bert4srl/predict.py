
import bert_utils as util
import json
import numpy as np
from typing import List
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, Dataset
from transformers import BertForTokenClassification, AdamW
from transformers import get_linear_schedule_with_warmup
import logging, sys
import time
from transformers import BertTokenizer
from tqdm import tqdm
import torch
from torch.nn import CrossEntropyLoss
from transformers import pipeline
from sklearn.model_selection import train_test_split
from torch.utils.data import SequentialSampler
import warnings
warnings.filterwarnings("ignore")
def predictions(batchsize):
    GPU_IX=0
    _, USE_CUDA = util.get_torch_device(GPU_IX)
    FILE_HAS_GOLD = True
    SEQ_MAX_LEN = 256
    BATCH_SIZE = batchsize
    # IMPORTANT NOTE: We predict on the dev set to make the results comparable with your previous models from this course
    TEST_DATA_PATH = "data\en_ewt-up-test.conllu" # "data/conll2003.dev.conll"
    # TEST_DATA_PATH = "data/trial_unk_data.conll"
    MODEL_DIR = "saved_models/MY_BERT_NER/"
    LOAD_EPOCH = 1
    INPUTS_PATH=f"{MODEL_DIR}/EPOCH_{LOAD_EPOCH}/model_inputs.txt"
    OUTPUTS_PATH=f"{MODEL_DIR}/EPOCH_{LOAD_EPOCH}/model_outputs.txt"
    PAD_TOKEN_LABEL_ID = CrossEntropyLoss().ignore_index # -100

    model, tokenizer = util.load_model(BertForTokenClassification, BertTokenizer, f"{MODEL_DIR}/EPOCH_{LOAD_EPOCH}")
    label2index = util.load_label_dict(f"{MODEL_DIR}/label2index.json")
    index2label = {v:k for k,v in label2index.items()}
    seq_lens = len(label2index)
    # prediction_inputs, prediction_masks, gold_labels = read_two(TESTFILE)


    X_test, y_test, labelindex, _, _ = util.read_json_srl(TESTFILE)
    test_inputs, test_masks, test_predicate_labels, test_labels, seq_lengths = util.data_to_tensors(X_test, 
                                                                                                tokenizer, 
                                                                                                max_len=SEQ_MAX_LEN, 
                                                                                                labels=y_test, 
                                                                                                label2index=labelindex,
                                                                                                pad_token_label_id=PAD_TOKEN_LABEL_ID)

    # Create the DataLoader for our training set.
    test_data = TensorDataset(test_inputs, test_masks, test_labels, test_predicate_labels)
    test_sampler = RandomSampler(test_data)
    prediction_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=BATCH_SIZE)

    print('Predicting labels for {:,} test sentences...'.format(len(prediction_inputs)))
    
    results, preds_list = util.evaluate_bert_model(prediction_dataloader, BATCH_SIZE, model, tokenizer, index2label, 
                                                        PAD_TOKEN_LABEL_ID, full_report=True, prefix="Test Set")
    print("  Test Loss: {0:.2f}".format(results['loss']))
    print("  Precision: {0:.2f} || Recall: {1:.2f} || F1: {2:.2f}".format(results['precision']*100, results['recall']*100, results['f1']*100))

    with open(OUTPUTS_PATH, "w") as fout:
        with open(INPUTS_PATH, "w") as fin:
            for sent, pred in preds_list:
                fin.write(" ".join(sent)+"\n")
                fout.write(" ".join(pred)+"\n")
