import torch
import torch.nn as nn
import torch.nn.functional as F

#-------------------------------------- Baseline Model --------------------------------------#

class Baseline(nn.Module):

    def __init__(self, embedding_dim, vocab):
        super(Baseline, self).__init__()

        self.embedding = nn.Embedding.from_pretrained(vocab.vectors)
        self.fc = nn.Linear(embedding_dim, 1)

    def forward(self, x, lens, lengths=None):
        
        # Chage to word vectors
        embedded = self.embedding(x)
        
        # Average all word vectors in the given setence
        average = embedded.mean(0) 
        
        # Linear layer
        output = self.fc(average).squeeze(1)

        return output


#-------------------------------------- CNN Model --------------------------------------#    
class CNN(nn.Module):
    def __init__(self, embedding_dim, vocab):
        super(CNN, self).__init__()
        
        self.embedding = nn.Embedding.from_pretrained(vocab.vectors)
        
        self.conv2 = nn.Conv2d(1,50,kernel_size=(2, embedding_dim))
        self.conv4 = nn.Conv2d(1,50,kernel_size=(4, embedding_dim))
        
        self.fc1 = nn.Linear(embedding_dim, 1)


    def forward(self, x, lens, lengths=None):
        
        # Change to word vectors
        embedded = self.embedding(x)
        
        # Rearange each sentence to become a matrix
        embedded = embedded.permute(1,0,2)
        embedded = embedded.unsqueeze(1)
        
        # CNN layers and pooling
        x_1 = (F.relu(self.conv2(embedded)))
        x_1,_ = torch.max(x_1,2)
        x_2 = (F.relu(self.conv4(embedded)))
        x_2,_ = torch.max(x_2,2)
        
        # Concatinate the two CNN layers
        x = torch.cat([x_1, x_2], 1).squeeze()
        
        # Linear layer
        x = self.fc1(x)
        
        return x
    
    
#-------------------------------------- RNN Model --------------------------------------# 
class RNN(nn.Module):
    def __init__(self, embedding_dim, vocab, hidden_dim):
        super(RNN, self).__init__()

        self.embedding = nn.Embedding.from_pretrained(vocab.vectors)
        
        self.rnn = nn.GRU(embedding_dim, hidden_dim)
        
        self.fc1 = nn.Linear(embedding_dim, 1)

    def forward(self, x, lens, lengths=None):

        # Change to word vectors
        x = self.embedding(x)
        
        # Make sure correct hidden layer is chosen by eliminating the padding 
        x = nn.utils.rnn.pack_padded_sequence(x, lens)
        
        # Take only the hidden layers from the RNN
        _, h = self.rnn(x)
        
        # Linear layer
        x = self.fc1(h)
        
        return x.squeeze()