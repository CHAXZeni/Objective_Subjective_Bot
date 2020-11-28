import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#----------------------------------------------------------------------------# 

# Splits the data into three sets training, validation and test where each set of data has an equal amout of each label within them.
def datasplit():
    # Load the TSV file into python
    all_data = pd.read_csv('./data/data.tsv', sep='\t', engine='python')

    # Seperate labels and text
    text = all_data.values[:,0]
    labels = all_data.values[:,1]
    
    # Split into training and verfication
    text_trn, text_vrf, labels_trn, labels_vrf = train_test_split(text, labels, train_size=0.64, random_state=seed, stratify=labels)
    
    # Create overfit dataset
    text_of, text_extra, labels_of, labels_extra = train_test_split(text_trn, labels_trn, train_size=50, random_state=seed, stratify=labels_trn)
    
    # Split verfication into validation and test
    text_vld, text_tst, labels_vld, labels_tst = train_test_split(text_vrf, labels_vrf, train_size=1600, random_state=seed, stratify=labels_vrf)
    
    
    # Combine the labels and texts into one 
    train_data = pd.DataFrame({'text': text_trn, 'label':labels_trn})
    valid_data = pd.DataFrame({'text': text_vld, 'label':labels_vld})
    test_data = pd.DataFrame({'text': text_tst, 'label':labels_tst})
    of_data = pd.DataFrame({'text': text_of, 'label':labels_of})
    
    # Save files
    train_data.to_csv('./data/train.tsv', sep='\t', index = False)
    valid_data.to_csv('./data/validation.tsv', sep='\t', index = False)
    test_data.to_csv('./data/test.tsv', sep='\t', index = False)
    of_data.to_csv('./data/overfit.tsv', sep='\t', index = False)
    

#----------------------------------------------------------------------------# 

def main():
    datasplit()