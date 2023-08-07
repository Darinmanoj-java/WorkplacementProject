import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
import nltk
import re
from nltk.corpus import stopwords 
from collections import Counter
import numpy as np

import json
import torch
from torch import nn

from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

nltk.download('stopwords')
stop_words = set(stopwords.words('english')) 
TOP_WORDS = 1000

# preprocessing
def preprocess(x_train,y_train,x_test,y_test):

    vocab_list = build_dictionary(x_train)
    label_mapping = build_mapping(vocab_list)
    return tokenisation(x_train, label_mapping), \
        output_encoding(y_train), \
            tokenisation(x_test, label_mapping), \
                output_encoding(y_test), \
                    vocab_list
def clean_string(s):
    # Remove all non-word characters (everything except numbers and letters)
    s = re.sub(r"[^\w\s]", '', s)
    # Replace all runs of whitespaces with no space
    s = re.sub(r"\s+", '', s)
    # replace digits with no space
    s = re.sub(r"\d", '', s)

    return s
def build_dictionary(sentences):
    
    vocab_list = []

    for sent in sentences:
        if isinstance(sent, float): # sent is nan
             continue
        for word in sent.lower().split():
            word = clean_string(word)
            if word not in stop_words and word != '':
                vocab_list.append(word)
    return vocab_list

def build_mapping(vocab_list, top_words=TOP_WORDS):

    vocab_counter = Counter(vocab_list)
    # sorting on the basis of most common words
    corpus_ = sorted(vocab_counter,key=vocab_counter.get,reverse=True)[:top_words]
    # creating a dict
    mapping = {w:i+1 for i,w in enumerate(corpus_)}
    
    return mapping
def tokenisation(sentences, mapping):
    tokens = []
    for sentence in sentences:
        if isinstance(sentence, float):
             tokens.append([])
             continue
        tokens.append([mapping[clean_string(word)] for word in sentence.lower().split() 
                                     if clean_string(word) in mapping.keys()])
    return tokens

def output_encoding(labels, label_tf={'negative':0, 'positive':1}):

    return [label_tf[label] for label in labels]
# Utility functions
def padding_(sentences, seq_len):
    features = np.zeros((len(sentences), seq_len),dtype=int)
    for ii, review in enumerate(sentences):
        if len(review) != 0:
            features[ii, -len(review):] = np.array(review)[:seq_len]
    return features.tolist()
def json_save(container, path):
    fp = open(path, "w")
    json.dump(container, fp)
    return
def json_load(path):
    fp = open(path, "r")
    return json.load(fp)
def torch_save_model(model, path):
    torch.save(model.state_dict(), path)
    return
def torch_load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model
# Model
class SentimentRNN(nn.Module):
    def __init__(self, num_layers, vocab_size, hidden_size, output_size, embedding_size, drop_prob=0.3):
        super(SentimentRNN,self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
            
        self.rnn = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        self.classification = nn.Sequential(
            nn.Dropout(drop_prob),
            nn.Linear(hidden_size, output_size),
        )
        
    def forward(self, x):
        embeddings = self.embedding(x)  # shape: B x S x Feature   since batch = True
        b, _ = self.rnn(embeddings)
        b = b[:, -1, :]                 # -1 means the last time step
        y = self.classification(b)      # [batch, 2]
        
        return y
# Training
def step(dataloader, model, criterion, optimizer, config):
    model.train()

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(config['device']), labels.to(config['device'])
        # Creating new variables for the hidden state, otherwise
        # we'd backprop through the entire training history
        
        model.zero_grad()
        outputs = model(inputs)
        
        # calculate the loss and perform backprop CrossEntropy 
        loss = criterion(outputs.squeeze(), labels.long())
        loss.backward()
        #`clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        nn.utils.clip_grad_norm_(model.parameters(), config['clip'])
        optimizer.step()
# Evaluation
# a function to count the number of correct output
# def CorCnt(probs, labels):
#     _, preds = torch.max(probs, dim=1) #
#     return torch.sum(preds == labels.squeeze()).item()

# def evaluate(dataloader, model, criterion, config):
#     model.eval()
#     losses = []
#     correct_cnt = 0.0
#     for inputs, labels in dataloader:

#             inputs, labels = inputs.to(config['device']), labels.to(config['device'])

#             outputs = model(inputs)
#             loss = criterion(outputs.squeeze(), labels.long())

#             losses.append(loss.item())
            
#             correct_cnt += CorCnt(outputs,labels)
#     return losses, correct_cnt
# load data
base_csv = 'Data/IMDB Dataset.csv'
df = pd.read_csv(base_csv)
X,y = df['review'].values, df['sentiment'].values
x_train_raw,x_test_raw,y_train_raw,y_test_raw = train_test_split(X,y,stratify=y)
print(f'shape of train data is {x_train_raw.shape}')
print(f'shape of test data is {x_test_raw.shape}') 
x_train,y_train,x_test,y_test,vocab = preprocess(x_train_raw,y_train_raw,x_test_raw,y_test_raw)

train_data = {}
test_data = {}

train_data['x'] = padding_(x_train,500)
test_data['x']= padding_(x_test,500)
train_data['y'] = y_train
test_data['y'] = y_test
json_save(train_data, "train_data_new2.json")
json_save(test_data, "test_data_new2.json")

# Main Function
config = {}

config['batch_size'] = 50
config['clip'] = 5
config['epochs'] = 5 
# config['device'] = "cuda"
config['lr'] = 0.001
train_data = json_load("train_data_new2.json")
test_data = json_load("test_data_new2.json")
x_train = np.array(train_data['x'])
y_train = np.array(train_data['y'])
x_test = np.array(test_data['x'])
y_test = np.array(test_data['y'])
# create Tensor datasets
train_data = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
test_data = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))

# dataloaders
train_loader = DataLoader(train_data, shuffle=True, batch_size=config['batch_size'])
test_loader = DataLoader(test_data, shuffle=True, batch_size=config['batch_size'])
vocab_size = TOP_WORDS + 1 #extra 1 for padding

# model_config = {"num_layers": 2, "embedding_size": 64, "output_size": 2, "hidden_size": 256, "vocab_size": vocab_size}

# model = SentimentRNN(model_config['num_layers'],
#                      model_config['vocab_size'],
#                      model_config['hidden_size'],
#                      model_config['output_size'], 
#                      model_config['embedding_size']).to(config['device'])