import os

if __name__ == "__main__":
    os.system("python train.py --train_path data\en_ewt-up-train.conllu --dev_path data\en_ewt-up-train.conllu --max_len 256 --batch_size 16 --epochs 8 --learning_rate 1e-5")
    os.system("python predict.py --model_path saved_models/BERT_SRL_256_16_8_1e-05 --test_path data\en_ewt-up-test.conllu")