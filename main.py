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

# Determines the accuracy of set of predictions.
def predictionaccuracy(Predictions, Labels):
    Accuracy = 0
    
    for i in range(len(Labels)):
        
        if (((Predictions[i].item() >= 0.5) and (Labels[i].item() == 1)) or 
            ((Predictions[i].item() < 0.5) and (Labels[i].item() == 0))):
            Accuracy += 1 / (len(Labels))
                   
    return Accuracy


#------------------------------------------------------------------------------------------------------------------------------------------------#

# Computes loss and accuracy for a given set of data. If train == True, gradients and computed and the model is trained.
def NNcomputationloop(model, data, optimizer, loss_fnc, train):
    
    runningloss = 0
    runningaccuracy = 0 
    batch_count = 0
    
    for batch in data:
        texts, text_lengths = batch.text
        labels = batch.label

        # Reset Gradient
        if train == True: optimizer.zero_grad()

        # Get Predictions
        outputs = model(texts, text_lengths)

        # Computing Loss
        loss = loss_fnc(input=outputs.squeeze(), target=labels.float())
        
        # Compute Gradients
        if train == True: loss.backward()
        
        # Gradient Step
        if train == True: optimizer.step()
        
        # Record Accuracy and Loss
        runningloss += float(loss)
        runningaccuracy += predictionaccuracy(outputs, labels)
        
        batch_count += 1
       

    return runningloss/batch_count, runningaccuracy/batch_count


#------------------------------------------------------------------------------------------------------------------------------------------------#

# Training loop for a neural network with a predefined model, optimizer, loss function and data sets.
def trainingloop(model, optimizer, loss_fnc, epochs, train_iter, valid_iter, test_iter, sample):
    
    # Create containers for data
    Tloss = []
    Tacc = []
    Vloss = []
    Vacc = []
    
    start = time.time()
    
    # Run the training
    for epoch in range(epochs):
        
        # Training Loop
        tloss, tacc = NNcomputationloop(model, train_iter, optimizer, loss_fnc, True)
        
        # Record Data
        Tloss.append(tloss)
        Tacc.append(tacc)
        
        # Validation Loop 
        if epoch%sample == 0:
            vloss, vacc = NNcomputationloop(model, valid_iter, optimizer, loss_fnc, False)
            Vloss.append(vloss)
            Vacc.append(vacc)
        
    end = time.time()
    
    # Time taken in seconds
    total_time = end - start
    
    # Find the test loss and accuracy
    test_loss, test_accuracy = NNcomputationloop(model, test_iter, optimizer, loss_fnc, False)
    
    # Display final results
    return Tloss, Tacc, Vloss, Vacc, total_time, test_accuracy, test_loss


#------------------------------------------------------------------------------------------------------------------------------------------------#

def main(args):
    
    #--------------------- Hyperparameters ---------------------# 
    batch_size = args.batch_size
    emb_dim = args.emb_dim
    epochs = args.epochs
    lr = args.lr
    model_type = args.model
    rnn_hidden_dim = args.rnn_hidden_dim
    sample = args.sample
        
    #--------------------- Data Proccessing ---------------------#   
    
    # Definning required fields for the set of data   
    TEXT = data.Field(sequential=True,lower=True, tokenize='spacy', include_lengths=True)
    LABELS = data.Field(sequential=False, use_vocab=False)
    
    # Importing datasets in the form of the fields defined above
    train_data, val_data, test_data = data.TabularDataset.splits(
                path='data/', train='train.tsv',
                validation='validation.tsv', test='test.tsv', format='tsv',
                skip_header=True, fields=[('text', TEXT), ('label', LABELS)])
    
    # Create batches of data and keep keep the sentence lengths in each batch roughly the same
    train_iter, val_iter, test_iter = data.BucketIterator.splits(
          (train_data, val_data, test_data), batch_sizes=(batch_size, batch_size, batch_size),
            sort_key=lambda x: len(x.text), device=None, sort_within_batch=True, repeat=False)
    
    # Create a vocab with all words that have been tokenized from the created datasets
    TEXT.build_vocab(train_data,val_data, test_data)
    
    # Loading the GloVe vector which contais the words vectors for the defined vocab
    TEXT.vocab.load_vectors(torchtext.vocab.GloVe(name='6B', dim=100))
    vocab = TEXT.vocab
    
    
    #--------------------- Model Initialization ---------------------# 
    if model_type == 'baseline': model = Baseline(emb_dim, vocab)
    elif model_type == 'cnn': model = RNN(emb_dim, vocab)
    elif model_type == 'rnn': model = RNN(emb_dim, vocab, rnn_hidden_dim)
    
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    loss_fnc = torch.nn.BCEWithLogitsLoss()
    
    
    #--------------------- Running Training ---------------------#  
    Tloss, Tacc, Vloss, Vacc, total_time, test_accuracy, test_loss = trainingloop(model, optimizer, loss_fnc, epochs, train_iter, val_iter, test_iter, sample)
    
    # Save the models when needed
    # torch.save(model, 'model_baseline.pt' )
    
    
    #--------------------- Display Results ---------------------# 
    # Training and Validation accuracy and loss plots as well as final values for each set and run time
    Display_Results(Tloss, Tacc, Vloss, Vacc, total_time, test_accuracy, test_loss)
    
    return True

#------------------------------------------------------------------------------------------------------------------------------------------------#
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--model', type=str, default='baseline',
                        help="Model type: baseline,rnn,cnn (Default: baseline)")
    parser.add_argument('--sample', type=int, default=1)
    parser.add_argument('--emb-dim', type=int, default=100)
    parser.add_argument('--rnn-hidden-dim', type=int, default=100)

    args = parser.parse_args()

    main(args)
    