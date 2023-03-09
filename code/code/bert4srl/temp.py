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
