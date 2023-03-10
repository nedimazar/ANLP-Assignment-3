import bert_utils as util
from torch.nn import CrossEntropyLoss
from transformers import BertForTokenClassification, BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, Dataset
import numpy as np
import argparse


def args_function():
    parser = argparse.ArgumentParser()
    parser.add_argument('-modelpath', '--model_path', help='Path to a fine tuned model', default="saved_models\BERT_SRL_256_32_8_1e-05")
    parser.add_argument('-devpath', '--test_path', help='Path to Dev Set', default="data\en_ewt-up-test.conllu")
    parser.add_argument('-epochs', '--epochs', help = "The correct epoch of the fine tuned model", type=int, default=8)
    parser.add_argument('-batchsize', '--batch_size', type=int, default=32)
    parser.add_argument('-learningrate', '--learning_rate', type=float, default=1e-5)
    parser.add_argument('-maxlen', '--max_len', type=int, default=256)
    args = parser.parse_args()
    return args


def predictions():
    args = args_function()
    SEQ_MAX_LEN = args.max_len
    BATCH_SIZE = args.batch_size
    TEST_DATA_PATH = args.test_path
    # TEST_DATA_PATH = "data/trial_unk_data.conll"
    MODEL_DIR = args.model_path
    LOAD_EPOCH = args.epochs
    INPUTS_PATH=f"{MODEL_DIR}/EPOCH_{LOAD_EPOCH}/model_inputs.txt"
    OUTPUTS_PATH=f"{MODEL_DIR}/EPOCH_{LOAD_EPOCH}/model_outputs.txt"
    PAD_TOKEN_LABEL_ID = CrossEntropyLoss().ignore_index # -100



    model, tokenizer = util.load_model(BertForTokenClassification, BertTokenizer, f"{MODEL_DIR}/EPOCH_{LOAD_EPOCH}")
    index2label = util.load_label_dict(f"label2index.json")
    label2index = {v:k for k,v in index2label.items()}
    seq_lens = len(label2index)

    data, label, labelindex, _, _ = util.read_json_srl(TEST_DATA_PATH)

    prediction_inputs, prediction_mask, prediction_predicate_labels, prediction_labels, prediction_seq_lengths = util.data_to_tensors(data, 
                                                                                                    tokenizer, 
                                                                                                    max_len=SEQ_MAX_LEN, 
                                                                                                    labels=label, 
                                                                                                    label2index=label2index,
                                                                                                    pad_token_label_id=PAD_TOKEN_LABEL_ID)

    prediction_data = TensorDataset(prediction_inputs, prediction_mask, prediction_labels, prediction_predicate_labels)
    prediction_sampler = RandomSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=BATCH_SIZE)



    print('Predicting labels for {:,} test sentences...'.format(len(prediction_inputs)))
    print("oindex", index2label, "label", label2index)
    # exit()
    results, preds_list = util.evaluate_bert_model(prediction_dataloader, BATCH_SIZE, model, tokenizer, label2index, 
                                                        PAD_TOKEN_LABEL_ID, full_report=True, prefix="Test Set")
    print("  Test Loss: {0:.2f}".format(results['loss']))
    print("  Precision: {0:.2f} || Recall: {1:.2f} || F1: {2:.2f}".format(results['precision']*100, results['recall']*100, results['f1']*100))

    with open(OUTPUTS_PATH, "w") as fout:
        with open(INPUTS_PATH, "w") as fin:
            for sent, pred in preds_list:
                fin.write(" ".join(sent)+"\n")
                fout.write(" ".join(pred)+"\n")


if __name__ == "__main__":
    predictions()