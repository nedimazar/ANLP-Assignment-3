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
from torch.utils.data import SequentialSampler
import warnings
warnings.filterwarnings("ignore")

PAD_TOKEN_LABEL_ID = CrossEntropyLoss().ignore_index # -100
torch.backends.cuda.max_split_size_bytes = 64 * 1024 * 1024
BERT_MODEL_NAME = 'bert-base-multilingual-cased'
SAVE_MODEL_DIR = "saved_models/MY_BERT_NER"
TESTFILE = "data\en_ewt-up-test.conllu"
TRAINFILE = "data\en_ewt-up-test.conllu"
EPOCHS = 4
GPU_RUN_IX=0
SEED_VAL = 1234500
SEQ_MAX_LEN = 256
GPU_IX=0
SEQLENGTH = 512

device, USE_CUDA = util.get_torch_device(GPU_IX)

PRINT_INFO_EVERY = 10 # Print status only every X batches
GRADIENT_CLIP = 1.0
LEARNING_RATE = 1e-5
BATCH_SIZE = 8
LABELS_FILENAME = f"{SAVE_MODEL_DIR}/label2index.json"
LOSS_TRN_FILENAME = f"{SAVE_MODEL_DIR}/Losses_Train_{EPOCHS}.json"
LOSS_DEV_FILENAME = f"{SAVE_MODEL_DIR}/Losses_Dev_{EPOCHS}.json"

def predictions():
    GPU_IX=0
    _, USE_CUDA = util.get_torch_device(GPU_IX)
    FILE_HAS_GOLD = True
    SEQ_MAX_LEN = 256
    BATCH_SIZE = 4
    # IMPORTANT NOTE: We predict on the dev set to make the results comparable with your previous models from this course
    TEST_DATA_PATH = "data/trial_mini_data.conll" # "data/conll2003.dev.conll"
    # TEST_DATA_PATH = "data/trial_unk_data.conll"
    MODEL_DIR = "saved_models/MY_BERT_NER/"
    LOAD_EPOCH = 1
    INPUTS_PATH=f"{MODEL_DIR}/EPOCH_{LOAD_EPOCH}/model_inputs.txt"
    OUTPUTS_PATH=f"{MODEL_DIR}/EPOCH_{LOAD_EPOCH}/model_outputs.txt"
    PAD_TOKEN_LABEL_ID = CrossEntropyLoss().ignore_index # -100

    console_hdlr = logging.StreamHandler(sys.stdout)
    file_hdlr = logging.FileHandler(filename=f"{MODEL_DIR}/EPOCH_{LOAD_EPOCH}/BERT_TokenClassifier_predictions.log")
    logging.basicConfig(level=print, handlers=[console_hdlr, file_hdlr])

    model, tokenizer = util.load_model(BertForTokenClassification, BertTokenizer, f"{MODEL_DIR}/EPOCH_{LOAD_EPOCH}")
    label2index = util.load_label_dict(f"{MODEL_DIR}/label2index.json")
    index2label = {v:k for k,v in label2index.items()}
    seq_lens = len(label2index)
    # prediction_inputs, prediction_masks, gold_labels = read_two(TESTFILE)


    prediction_data, prediction_sampler, prediction_dataloader, index2label = get_data(TESTFILE)


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



def preprocess_data_for_bert(all_sentences, labels, label_dict, all_preds, max_length = 128):
    # Create a binary sequence per instance indicating which predicate is currently labeling
    binary_labels = []
    for i, seq_labels in enumerate(labels):
        binary_seq_labels = []
        for j, label in enumerate(seq_labels):
            if label == 'V':
                binary_seq_labels.append(1)
            else:
                binary_seq_labels.append(0)
        # make sure the length of the sequence is the same as the sentence through paddinf
        binary_labels.append(binary_seq_labels)
    
    # Tokenize the sentences and map the labels to their corresponding label ids
    tokenized_sentences = []
    tokenized_labels = []
    for i, sentence in enumerate(all_sentences):
        tokenized_sentence = []
        tokenized_label = []
        for j, word in enumerate(sentence):
            tokenized_word = tokenizer.tokenize(word)
            tokenized_sentence.extend(tokenized_word)
            if len(tokenized_word) == 1:
                try:
                    tokenized_label.append(label_dict[labels[i][j]])
                except:
                    tokenized_label.append(label_dict["_"])
            else:
                try:
                    tokenized_label.extend([label_dict[labels[i][j]]])
                except:
                    tokenized_label.extend([label_dict["_"]])

        # make sure the length of the sentence is not longer than the max length trhough padding

        rest_length_label = max_length - len(tokenized_label)
        tokenized_label += [0] * rest_length_label
        tokenized_sentences.append(tokenized_sentence)
        tokenized_labels.append(tokenized_label)
        
    # Convert the tokenized sentences and labels to input features and targets for BERT
    
    
    input_features = []
    targets = []
    for i, sentence in enumerate(tokenized_sentences):
        sentence = tokenizer.convert_tokens_to_ids(sentence)
        binary_label = binary_labels[i]
        sentence.extend(binary_label)
        attention_mask_1 = len(sentence)
        
        padding_length = max_length - len(sentence)
        
        sentence += [0]*padding_length
    

        attentionmask = [1]*attention_mask_1 + [0]*padding_length
        input_features.append({
            'input_ids': sentence,
            'attention_mask': attentionmask,
            'labels': tokenized_labels[i]
        })
        # print(tokenized_labels, i)
    

    print("input_features", len(input_features), np.unique([len(f['input_ids']) for f in input_features]))
    return input_features



def get_data(file_name):

    all_sentences, labels, label_dict, all_preds, _ = util.read_json_srl(file_name)
    
    input_features = preprocess_data_for_bert(all_sentences[:100], labels[:100], label_dict, all_preds[:100], max_length = SEQLENGTH)
    
    return input_features,label_dict

def toTensor(train_data):
    
    input_ids = torch.tensor([f['input_ids'] for f in train_data], dtype=torch.long)
    masks = torch.tensor([f['attention_mask'] for f in train_data], dtype=torch.long)
    labels = torch.tensor([f['labels'] for f in train_data], dtype=torch.long)
    
    return input_ids, masks, labels

# function that reads the util code
if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME, do_basic_tokenize=False)
    
    # Load the training data
    train_input_features, index2label = get_data(TRAINFILE)
    train_inputs, train_masks, train_labels = toTensor(train_input_features)
    
    # Create the DataLoader for our training set.
    
    train_data = TensorDataset(train_inputs, train_masks, torch.tensor(train_labels))
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)


    # Load the validation data
    dev_input_features, index2labeldev = get_data(TESTFILE)
    dev_inputs, dev_masks, dev_labels = toTensor(dev_input_features)
    # Create the DataLoader for our validation set.
    dev_data = TensorDataset(dev_inputs, dev_masks, torch.tensor(dev_labels))
    dev_sampler = RandomSampler(dev_data)
    dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=BATCH_SIZE)
    
    util.save_label_dict(index2label, filename=LABELS_FILENAME)

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

    for epoch_i in (range(1, EPOCHS+1)):
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
        results, preds_list = util.evaluate_bert_model(dev_dataloader, BATCH_SIZE, model, tokenizer, index2label, PAD_TOKEN_LABEL_ID, prefix="Validation Set")
        loss_dev_values.append(results['loss'])
        print("  Validation Loss: {0:.2f}".format(results['loss']))
        print("  Precision: {0:.2f} || Recall: {1:.2f} || F1: {2:.2f}".format(results['precision']*100, results['recall']*100, results['f1']*100))
        print("  Validation took: {:}".format(util.format_time(time.time() - t0)))


        # Save Checkpoint for this Epoch
        util.save_model(f"{SAVE_MODEL_DIR}/EPOCH_{epoch_i}", {"args":[]}, model, tokenizer)


    util.save_losses(loss_trn_values, filename=LOSS_TRN_FILENAME)
    util.save_losses(loss_dev_values, filename=LOSS_DEV_FILENAME)
    print("")
    print("Training complete!")


    # predictions()