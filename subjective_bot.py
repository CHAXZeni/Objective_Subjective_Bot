import torch
import torch.optim as optim

import torchtext
from torchtext import data
import spacy

import argparse
import os

import time

from models import *
from displaytools import *
#------------------------------------------------------------------------------------------------------------------------------------------------#
    
# Determine wether or not a setence is subjective or objective using the different models created.
def subjective_bot():
    while True:
        print('Enter a Sentence\n')
        sentence = input()
        
        # Create vocab
        
        # Definning required fields for the set of data   
        TEXT = data.Field(sequential=True,lower=True, tokenize='spacy', include_lengths=True)
        LABELS = data.Field(sequential=False, use_vocab=False)
    
        # Importing datasets in the form of the fields defined above
        train_data, val_data, test_data = data.TabularDataset.splits(
                    path='data/', train='train.tsv',
                    validation='validation.tsv', test='test.tsv', format='tsv',
                    skip_header=True, fields=[('text', TEXT), ('label', LABELS)])
        
        # Create a vocab with all words that have been tokenized from the created datasets
        TEXT.build_vocab(train_data,val_data, test_data)
        
        # Loading the GloVe vector which contais the words vectors for the defined vocab
        TEXT.vocab.load_vectors(torchtext.vocab.GloVe(name='6B', dim=100))
        vocab = TEXT.vocab
        
        # Load Models
        model_baseline = torch.load('model_baseline.pt')
        model_cnn = torch.load('model_cnn.pt')
        model_rnn = torch.load('model_rnn.pt')

        # Convert words to tokens and then into integers
        spacy_en = spacy.load('en')
        tokens = [tok.text for tok in spacy_en(sentence)]
        token_ints = [vocab.stoi[tok] for tok in tokens]

        # Convert to tensor
        token_tensor = torch.LongTensor(token_ints).view(-1,1) # Shape is [sentence_len, 1]


        # Length of tensor
        lengths = torch.Tensor([len(token_ints)])
        
        # Get predictions
        baseline_prediction = F.sigmoid(model_baseline(token_tensor, lengths))
        cnn_prediction = F.sigmoid(model_cnn(token_tensor, lengths))
        rnn_prediction = F.sigmoid(model_rnn(token_tensor, lengths))


        print('Model basline: (%.3f)\nModel cnn: (%.3f)\nModel rnn: (%.3f)\n' 
              % (baseline_prediction.item(), cnn_prediction.item(), rnn_prediction.item())) 
        
subjective_bot()
        

