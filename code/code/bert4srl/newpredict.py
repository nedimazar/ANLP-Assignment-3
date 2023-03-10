import bert_utils as util
from torch.nn import CrossEntropyLoss
from transformers import BertForTokenClassification, BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, Dataset


def predictions():
    GPU_IX=0
    _, USE_CUDA = util.get_torch_device(GPU_IX)
    FILE_HAS_GOLD = True
    SEQ_MAX_LEN = 256
    BATCH_SIZE = 4
    # IMPORTANT NOTE: We predict on the dev set to make the results comparable with your previous models from this course
    TEST_DATA_PATH = "data/en_ewt-up-test.conllu" # "data/conll2003.dev.conll"
    # TEST_DATA_PATH = "data/trial_unk_data.conll"
    MODEL_DIR = "saved_models/BERT_SRL_256_32_16_1e-05"
    LOAD_EPOCH = 15
    INPUTS_PATH=f"{MODEL_DIR}/EPOCH_{LOAD_EPOCH}/model_inputs.txt"
    OUTPUTS_PATH=f"{MODEL_DIR}/EPOCH_{LOAD_EPOCH}/model_outputs.txt"
    PAD_TOKEN_LABEL_ID = CrossEntropyLoss().ignore_index # -100



    model, tokenizer = util.load_model(BertForTokenClassification, BertTokenizer, f"{MODEL_DIR}/EPOCH_{LOAD_EPOCH}")
    label2index = util.load_label_dict(f"{MODEL_DIR}/label2index.json")
    index2label = {v:k for k,v in label2index.items()}
    seq_lens = len(label2index)
    # prediction_inputs, prediction_masks, gold_labels = read_two(TESTFILE)

    data, label, labelindex, _, _ = util.read_json_srl(TEST_DATA_PATH)
    prediction_inputs, prediction_mask, prediction_predicate_labels, prediction_labels, prediction_seq_lengths = util.data_to_tensors(data[:50], 
                                                                                                    tokenizer, 
                                                                                                    max_len=SEQ_MAX_LEN, 
                                                                                                    labels=label[:50], 
                                                                                                    label2index=labelindex,
                                                                                                    pad_token_label_id=PAD_TOKEN_LABEL_ID)
    print(prediction_labels[1])
    prediction_data = TensorDataset(prediction_inputs, prediction_mask, prediction_labels, prediction_predicate_labels)
    prediction_sampler = RandomSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=BATCH_SIZE)



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


if __name__ == "__main__":
    predictions()