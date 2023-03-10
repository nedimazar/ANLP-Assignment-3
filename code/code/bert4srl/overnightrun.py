import os

if __name__ == "__main__":
    os.system("python train.py -epochs 6 -batchsize 8 -learningrate 1e-5 -maxlen 256")
    os.system("python train.py -epochs 8 -batchsize 16 -learningrate 1e-5 -maxlen 256")
    os.system("python train.py -epochs 16 -batchsize 32 -learningrate 1e-5 -maxlen 256")
    os.system("python train.py -epochs 16 -batchsize 64 -learningrate 1e-5 -maxlen 256")
    os.system("python train.py -epochs 16 -batchsize 128 -learningrate 1e-5 -maxlen 256")
    os.system("python train.py -epochs 8 -batchsize 32 -learningrate 1e-4 -maxlen 256")
    os.system("python train.py -epochs 8 -batchsize 16 -learningrate 1e-4 -maxlen 256")
