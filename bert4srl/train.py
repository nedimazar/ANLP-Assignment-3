import bert_utils as util
import json
import numpy as np
from typing import List
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, Dataset
from transformers import BertForTokenClassification, AdamW
from transformers import get_linear_schedule_with_warmup
import logging, sys
import time
import random
from transformers import BertTokenizer
from tqdm import tqdm
import torch
from torch.nn import CrossEntropyLoss
from transformers import pipeline
# from predict import predictions
from sklearn.model_selection import train_test_split
from torch.utils.data import SequentialSampler
import argparse
import warnings
# import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

import os
PAD_TOKEN_LABEL_ID = CrossEntropyLoss().ignore_index # -100
torch.backends.cuda.max_split_size_bytes = 64 * 1024 * 1024
BERT_MODEL_NAME = 'bert-base-multilingual-cased'
# SAVE_MODEL_DIR = "saved_models/MY_BERT_NER"
TESTFILE = "data\en_ewt-up-test.conllu"
TRAINFILE = "data\en_ewt-up-train.conllu"
EPOCHS = 10
GPU_RUN_IX=0
SEED_VAL = 2332
SEQ_MAX_LEN = 256
GPU_IX=0
SEQLENGTH = 512

device, USE_CUDA = util.get_torch_device(GPU_IX)

PRINT_INFO_EVERY = 10 # Print status only every X batches
GRADIENT_CLIP = 1.0
LEARNING_RATE = 1e-5
BATCH_SIZE = 16


def args_function():
    parser = argparse.ArgumentParser()
    parser.add_argument('-trainpath', '--train_path', help='Path to Train set', default="data\en_ewt-up-train.conllu")
    parser.add_argument('-devpath', '--dev_path', help='Path to Dev Set', default="data\en_ewt-up-test.conllu")
    parser.add_argument('-epochs', '--epochs', type=int, default=12)
    parser.add_argument('-batchsize', '--batch_size', type=int, default=32)
    parser.add_argument('-learningrate', '--learning_rate', type=float, default=1e-5)
    parser.add_argument('-maxlen', '--max_len', type=int, default=256)
    args = parser.parse_args()
    return args

# function that reads the util code
if __name__ == "__main__":
    args = args_function()
    SEQ_MAX_LEN = args.max_len
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    TRAINFILE = args.train_path
    TESTFILE = args.dev_path
    LEARNING_RATE = args.learning_rate
    SAVE_MODEL_DIR =f"saved_models/BERT_SRL_{args.max_len}_{args.batch_size}_{args.epochs}_{args.learning_rate}"
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME, do_basic_tokenize=False)
    try:
        os.makedirs(SAVE_MODEL_DIR)
    except:
        SAVE_MODEL_DIR += "_"+str(random.randint(0, 11000))
    LABELS_FILENAME = f"label2index.json"
    LOSS_TRN_FILENAME = f"{SAVE_MODEL_DIR}/Losses_Train_{EPOCHS}.json"
    LOSS_DEV_FILENAME = f"{SAVE_MODEL_DIR}/Losses_Dev_{EPOCHS}.json"
    # Load the training data With Predicate labels
    data, label, labelindex, _, _ = util.read_json_srl(TRAINFILE)
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.1, random_state=42)
    train_inputs, train_masks, train_predicate_labels, train_labels, seq_lengths = util.data_to_tensors(X_train, 
                                                                                                tokenizer, 
                                                                                                max_len=SEQ_MAX_LEN, 
                                                                                                labels=y_train, 
                                                                                                label2index=labelindex,
                                                                                                pad_token_label_id=PAD_TOKEN_LABEL_ID)

    # Create the DataLoader for our training set.
    train_data = TensorDataset(train_inputs, train_masks, train_labels, train_predicate_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)


    dev_inputs, dev_masks,dev_predicate_labels,  dev_labels, dev_lens = util.data_to_tensors(X_test, 
                                                                    tokenizer, 
                                                                    max_len=SEQ_MAX_LEN, 
                                                                    labels=y_test, 
                                                                    label2index=labelindex,
                                                                    pad_token_label_id=PAD_TOKEN_LABEL_ID)

    # Create the DataLoader for our Development set.
    dev_data = TensorDataset(dev_inputs, dev_masks, dev_labels, dev_predicate_labels)
    dev_sampler = RandomSampler(dev_data)
    dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=BATCH_SIZE)


    labelindextemp = {value: key for key, value in labelindex.items()}
    loss_list = []
    util.save_label_dict(labelindextemp, filename=LABELS_FILENAME)

    model = BertForTokenClassification.from_pretrained(BERT_MODEL_NAME, num_labels=len(labelindex))
    model.config.finetuning_task = 'token-classification'
    model.config.id2label = labelindextemp
    model.config.label2id = labelindex
    if USE_CUDA: model.cuda()

    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_dataloader) * EPOCHS

    # Create optimizer and the learning rate scheduler.
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,
                                            num_training_steps=total_steps)
    
    loss_trn_values, loss_dev_values = [], []

    for epoch_i in (range(1, EPOCHS+1)):
        epochplot = []

        # Perform one full pass over the training set.
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i, EPOCHS))
        print('Training...')

        t0 = time.time()
        total_loss = 0
        model.train()

        # For each batch of training data...
        for step, batch in (enumerate(train_dataloader)):
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            model.zero_grad()

            # Perform a forward pass (evaluate the model on this training batch).
            outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs[0]
            total_loss += loss.item()


            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)

            # Update parameters
            optimizer.step()
            scheduler.step()
            loss_list.append(loss.item())
            epochplot.append(loss.item())
            # Progress update
            if step % PRINT_INFO_EVERY == 0 and step != 0:
                # Calculate elapsed time in minutes.
                elapsed = util.format_time(time.time() - t0)
                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.    Loss: {}.'.format(step, len(train_dataloader),
                                                                                                elapsed, loss.item()))

        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader)

        # Store the loss value for plotting the learning curve.
        loss_trn_values.append(avg_train_loss)
        print("")
        print("  Average training loss: {0:.4f}".format(avg_train_loss))
        print("  Training Epoch took: {:}".format(util.format_time(time.time() - t0)))

        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on our validation set.
        t0 = time.time()

        results, preds_list = util.evaluate_bert_model(dev_dataloader, BATCH_SIZE, model, tokenizer, labelindex, PAD_TOKEN_LABEL_ID, prefix="Validation Set")
        loss_dev_values.append(results['loss'])
        print("  Validation Loss: {0:.2f}".format(results['loss']))
        print("  Precision: {0:.2f} || Recall: {1:.2f} || F1: {2:.2f}".format(results['precision']*100, results['recall']*100, results['f1']*100))
        print("  Validation took: {:}".format(util.format_time(time.time() - t0)))
        util.save_model(f"{SAVE_MODEL_DIR}/EPOCH_{epoch_i}", {"args":[]}, model, tokenizer)

        # util.plot_loss(epochplot, filename=f"{SAVE_MODEL_DIR}/EPOCH_{epoch_i}_loss.png")
        # Save Checkpoint for this Epoch

        # util.save_model(f"{SAVE_MODEL_DIR}/EPOCH_{epoch_i}", {"args":[]}, model, tokenizer)

    ## Use matplotlip to plot the loss curve
    # save los as in text file
    file = open(f"{SAVE_MODEL_DIR}/loss.txt", "w")
    file.write(str(loss_list))
    file.close()
    
    # util.plot_loss(loss_list)
    util.save_losses(loss_trn_values, filename=LOSS_TRN_FILENAME)
    util.save_losses(loss_dev_values, filename=LOSS_DEV_FILENAME)
    print("")
    print("Training complete!")

