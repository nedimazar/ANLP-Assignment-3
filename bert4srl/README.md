## README
### Overview
This code defines a BERT-based named entity recognition (NER) model using the transformers package.

### The code is organized in the following sections:


### Usage
`Python version: 3.8`
To run this code, first, install the required packages listed in the requirements.txt file. Then, you can train and evaluate the NER model by running the train.py script. To see the available options for the train.py script, you can run python train.py --help.

The code assumes that the input data is in the CoNLL-U format. The read_json_srl() function in the bert_utils.py file is used to parse the CoNLL-U formatted data.

To train the model:

python train.py

To use the trained model for prediction

python predict.py

### Code structure
train.py
This script defines the main function for training the NER model. It first loads the data and then tokenizes it using the BERT tokenizer. It then creates data loaders for the training and validation data. The BERT-based NER model is defined using the BertForTokenClassification class from the transformers package. The model is trained using the train() function, and the model's performance is evaluated using the evaluate() function. Finally, the trained model is saved to disk.

bert_utils.py
This file defines various utility functions for the NER model. These include:

read_json_srl(): Parses CoNLL-U formatted data.
data_to_tensors(): Tokenizes and converts input data to PyTorch tensors.
save_label_dict(): Saves the mapping between label indices and labels to disk.
get_torch_device(): Returns the appropriate device for running PyTorch on (CPU or GPU).
predict.py
This file defines a function for making predictions on new data using the trained NER model.

requirements.txt
This file lists the required packages and their versions for running the code.

### Note
We had some problems fixing dependency issues regarding `matplotlib`, so we commented out the lines that generate plots in the various scripts. Feel free to install `matplotlib` and try running them.

