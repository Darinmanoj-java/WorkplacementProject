import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
# from torchtext.data import Field, Example, Dataset, BucketIterator
from torchtext.data import Field, Example, Dataset
from torch.utils.data import DataLoader, random_split

# Set random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# Define the Fields
Text=Field(tokenize='spacy', lower=True, include_lengths=True)
Label=Field(dtype=torch.String)

# Load the data
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    examples = []
    for i, row in df.iterrows():
        text = row['review']
        label = row['sentiment']
        examples.append(Example.fromlist([text, label], fields=[('text', Text), ('label', Label)]))
    return examples

# Create Dataset and Fields
train_examples = load_data('IMDB Dataset.csv')
train_len = int(len(train_examples) * 0.7)
test_len = int(len(train_examples) * 0.15)

train_dataset, valid_dataset, test_dataset = random_split(
    train_examples, [train_len, test_len, len(train_examples) - train_len - test_len]
)

# Build the vocabulary
Text.build_vocab(train_dataset,max_size=25000,vectors='glove.6B.100d', unk_init=torch.Tensor.normal_, min_freq=3)
Label.build_vocab(train_dataset)

# Create DataLoaders for the data
BATCH_SIZE = 64

train_iterator = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_iterator = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_iterator = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
