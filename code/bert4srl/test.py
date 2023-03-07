import bert_utils as util
import json
import numpy as np
from typing import List
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from transformers import BertForTokenClassification, AdamW
from transformers import get_linear_schedule_with_warmup
import logging, sys
import time
from transformers import BertTokenizer
from tqdm import tqdm
import torch
from torch.nn import CrossEntropyLoss


PAD_TOKEN_LABEL_ID = CrossEntropyLoss().ignore_index # -100

BERT_MODEL_NAME = 'bert-base-multilingual-cased'
SAVE_MODEL_DIR = "saved_models/MY_BERT_NER/"
BATCH_SIZE = 4
EPOCHS = 2
GPU_RUN_IX=0
SEED_VAL = 1234500
SEQ_MAX_LEN = 256
GPU_IX=0

device, USE_CUDA = util.get_torch_device(GPU_IX)

PRINT_INFO_EVERY = 10 # Print status only every X batches
GRADIENT_CLIP = 1.0
LEARNING_RATE = 1e-5
BATCH_SIZE = 4
LABELS_FILENAME = f"{SAVE_MODEL_DIR}/label2index.json"
LOSS_TRN_FILENAME = f"{SAVE_MODEL_DIR}/Losses_Train_{EPOCHS}.json"
LOSS_DEV_FILENAME = f"{SAVE_MODEL_DIR}/Losses_Dev_{EPOCHS}.json"

class InputFeatures(object):
    def __init__(self, input_ids, attention_mask, token_type_ids, label_ids):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label_ids = label_ids

def convert_to_features(data, tokenizer, label_dict, max_seq_length):
    """Converts a list of dictionaries to a list of InputFeatures."""

    features = []

    for ins in tqdm(data):
        # Create binary sequence to indicate which predicate is being labeled
        for instance in ins:
            # print(instance)
            predicate_idx = instance['pred_sense'][0]
            predicate_mask = [1 if i == predicate_idx else 0 for i in range(len(instance['seq_words']))]

            # Tokenize the input sentence and get the token-to-character alignments
            tokens = []
            char_to_word_offset = []
            for i, word in enumerate(instance['seq_words']):
                word_tokens = tokenizer.tokenize(word)
                tokens.extend(word_tokens)
                for j in range(len(word_tokens)):
                    char_to_word_offset.append(i)

            # Truncate tokens and character offsets to the maximum sequence length
            if len(tokens) > max_seq_length - 2:
                tokens = tokens[:(max_seq_length - 2)]
                char_to_word_offset = char_to_word_offset[:(max_seq_length - 2)]

            # Add special tokens and create input ids, attention mask, and token type ids
            tokens = ['[CLS]'] + tokens + ['[SEP]']
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            attention_mask = [1] * len(input_ids)
            token_type_ids = [0] * len(tokens)

            # Pad input ids, attention mask, and token type ids with zeros
            padding_length = max_seq_length - len(input_ids)
            input_ids += [0] * padding_length
            attention_mask += [0] * padding_length
            token_type_ids += [0] * padding_length
            predicate_mask += [0] * padding_length

            # Create InputFeatures object and append to features list
            # label_ids = instance['bio']
            label_padding_length = max_seq_length - len(instance['bio'])
            label_ids = [label_dict[el] for el in instance['bio']]
            label_ids += [0] * label_padding_length
            # for el in instance['bio']:
                # label_ids.append(label_dict[el])
            features.append(InputFeatures(input_ids=input_ids, attention_mask=attention_mask,
                                          token_type_ids=token_type_ids, label_ids=label_ids))

    return features


def get_data(file_name):

    all_sentences, labels, label_dict, all_preds, chunked = util.read_json_srl(file_name)
    max_length = (max([len(x) for x in all_sentences]))
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)


    train_features = convert_to_features(chunked, tokenizer, label_dict, max_length)
    # print(features[0].input_ids, features[0].attention_mask, features[0].token_type_ids, features[0].label_ids)
    
    # Initialize Tokenizer
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME, do_basic_tokenize=False)

    # train_features = convert_to_features(train_data, tokenizer, train_label2index, SEQ_MAX_LEN)
    train_inputs = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    train_masks = torch.tensor([f.attention_mask for f in train_features], dtype=torch.long)
    train_labels = torch.tensor([f.label_ids for f in train_features], dtype=torch.long)
    seq_lengths = [max_length for f in train_features]

    # util.save_label_dict(label_dict, filename=LABELS_FILENAME)
    index2label = {v: k for k, v in label_dict.items()}

    # Create the DataLoader for our training set.
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)
    
    return train_data, train_sampler, train_dataloader, index2label
# function that reads the util code
if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME, do_basic_tokenize=False)
 
    train_data, train_sampler, train_dataloader, index2label = get_data("/Users/jonas/Desktop/Dutsy/ANLP-Assignment-3/code/bert4srl/data/en_ewt-up-test.conllu")
    
    dev_data, dev_sampler, dev_dataloader, index2label = get_data("/Users/jonas/Desktop/Dutsy/ANLP-Assignment-3/code/bert4srl/data/en_ewt-up-test.conllu")


    model = BertForTokenClassification.from_pretrained(BERT_MODEL_NAME, num_labels=len(index2label))
    model.config.finetuning_task = 'token-classification'
    model.config.id2label = index2label
    model.config.label2id = index2label
    if USE_CUDA: model.cuda()

    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_dataloader) * EPOCHS

    # Create optimizer and the learning rate scheduler.
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,
                                            num_training_steps=total_steps)
    
    
    
    loss_trn_values, loss_dev_values = [], []


    for epoch_i in tqdm(range(1, EPOCHS+1)):
        # Perform one full pass over the training set.
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i, EPOCHS))
        print('Training...')

        t0 = time.time()
        total_loss = 0
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
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

            # Progress update
            if step % PRINT_INFO_EVERY == 0 and step != 0:
                # Calculate elapsed time in minutes.
                elapsed = util.format_time(time.time() - t0)
                # Report progress.
                logging.info('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.    Loss: {}.'.format(step, len(train_dataloader),
                                                                                                elapsed, loss.item()))

        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader)

        # Store the loss value for plotting the learning curve.
        loss_trn_values.append(avg_train_loss)

        logging.info("")
        logging.info("  Average training loss: {0:.4f}".format(avg_train_loss))
        logging.info("  Training Epoch took: {:}".format(util.format_time(time.time() - t0)))

        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on our validation set.
        t0 = time.time()
        results, preds_list = util.evaluate_bert_model(dev_dataloader, BATCH_SIZE, model, tokenizer, index2label, PAD_TOKEN_LABEL_ID, prefix="Validation Set")
        loss_dev_values.append(results['loss'])
        logging.info("  Validation Loss: {0:.2f}".format(results['loss']))
        logging.info("  Precision: {0:.2f} || Recall: {1:.2f} || F1: {2:.2f}".format(results['precision']*100, results['recall']*100, results['f1']*100))
        logging.info("  Validation took: {:}".format(util.format_time(time.time() - t0)))


        # Save Checkpoint for this Epoch
        util.save_model(f"{SAVE_MODEL_DIR}/EPOCH_{epoch_i}", {"args":[]}, model, tokenizer)


    util.save_losses(loss_trn_values, filename=LOSS_TRN_FILENAME)
    util.save_losses(loss_dev_values, filename=LOSS_DEV_FILENAME)
    logging.info("")
    logging.info("Training complete!")
