# def convert_to_features(data, tokenizer, label_dict, max_seq_length):
#     """Converts a list of dictionaries to a list of InputFeatures."""

#     features = []

#     for ins in tqdm(data):
#         # Create binary sequence to indicate which predicate is being labeled
#         for instance in ins:
#             # print(instance)
#             predicate_idx = instance['pred_sense'][0]
#             predicate_mask = [1 if i == predicate_idx else 0 for i in range(len(instance['seq_words']))]
#             print(predicate_mask)
#             # Tokenize the input sentence and get the token-to-character alignments
#             tokens = []
#             char_to_word_offset = []
#             for i, word in enumerate(instance['seq_words']):
#                 word_tokens = tokenizer.tokenize(word)
#                 tokens.extend(word_tokens)
#                 for j in range(len(word_tokens)):
#                     char_to_word_offset.append(i)

#             # Truncate tokens and character offsets to the maximum sequence length
#             if len(tokens) > max_seq_length - 2:
#                 tokens = tokens[:(max_seq_length - 2)]
#                 char_to_word_offset = char_to_word_offset[:(max_seq_length - 2)]

#             # Add special tokens and create input ids, attention mask, and token type ids
#             tokens = ['[CLS]'] + tokens + ['[SEP]']
#             input_ids = tokenizer.convert_tokens_to_ids(tokens)
#             attention_mask = [1] * len(input_ids)
#             token_type_ids = [0] * len(tokens)

#             # Pad input ids, attention mask, and token type ids with zeros
#             padding_length = max_seq_length - len(input_ids)
#             input_ids += [0] * padding_length
#             attention_mask += [0] * padding_length
#             token_type_ids += [0] * padding_length
#             predicate_mask += [0] * padding_length

#             # Create InputFeatures object and append to features list
#             # label_ids = instance['bio']
#             label_padding_length = max_seq_length - len(instance['bio'])
#             label_ids = [label_dict[el] for el in instance['bio']]
#             label_ids += [0] * label_padding_length
#             # for el in instance['bio']:
#                 # label_ids.append(label_dict[el])
#             features.append(InputFeatures(input_ids=input_ids, attention_mask=attention_mask,
#                                           token_type_ids=token_type_ids, label_ids=label_ids))

#     return features


import torch
from typing import List, Tuple, Dict

def preprocess_data_for_bert(all_sentences: List[List[str]], labels: List[List[str]],
                             label_dict: Dict[str, int], all_preds: List[Tuple[int, str]]) -> Tuple[List, List]:
    
    # Create a binary sequence per instance indicating which predicate is currently labeling
    binary_labels = []
    for i, seq_labels in enumerate(labels):
        binary_seq_labels = []
        for j, label in enumerate(seq_labels):
            if label == '_':
                binary_seq_labels.append(0)
            elif (i, all_preds[i][0]) == (j, label_dict[label]):
                binary_seq_labels.append(1)
            else:
                binary_seq_labels.append(0)
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
                tokenized_label.append(label_dict[labels[i][j]])
            else:
                tokenized_label.extend([label_dict[labels[i][j]]] * len(tokenized_word))
        tokenized_sentences.append(tokenized_sentence)
        tokenized_labels.append(tokenized_label)
    
    # Convert the tokenized sentences and labels to input features and targets for BERT
    input_features = []
    targets = []
    for i, sentence in enumerate(tokenized_sentences):
        input_ids = tokenizer.convert_tokens_to_ids(sentence)
        attention_mask = [1] * len(input_ids)
        input_features.append({
            'input_ids': input_ids,
            'attention_mask': attention_mask
        })
        targets.append(torch.tensor(tokenized_labels[i], dtype=torch.long))
    
    return input_features, targets, binary_labels


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