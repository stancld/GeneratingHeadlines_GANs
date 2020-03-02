"""
GANs for Abstractive Text Summarization
Project for Statistical Natural Language Processing (COMP0087)
University College London

File: Attention_seq2seq.py

Description of our model:

Collaborators:
    - Daniel Stancl
    - Dorota Jagnesakova
    - Guoliang HE
    - Zakhar Borok`
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class generator:
    """
    """
    def __init__(self, model = _Seq2Seq, loss_function = nn.NLL,
                 optimiser = optim.Adam, batch_size = 128, 
                 text_dictionary = text_dictionary, embeddings = pre_train_weight, 
                 **kwargs):
        """
        :param model:
            type:
            description:
        :loss_function:
            type:
            description:
        :optimiser:
            type:
            description:
        :batch_size:
            type:
            description:
        :text_dictionary:
            type:
            description:
        :embeddings:
            type:
            description:
        """
        # store some essential parameters and objects
        self.batch_size = batch_size
        self.text_dictionary = text_dictionary
        self.embeddings = embeddings
        
        ###---###
        self.grid = {'max_epochs': kwargs['max_epochs'],
                     'learning_rate': kwargs['learning_rate'],
                     'clip': kwargs['clip'],
                     # during training
                     'teacher_forcing_ratio': kwargs['teacher_forcing_ratio']
                     }
        OUTPUT_DIM = kwargs['OUTPUT_DIM']
        ENC_EMB_DIM = kwargs['ENC_EMB_DIM']
        #DEC_EMB_DIM = 1
        ENC_HID_DIM = kwargs['ENC_HID_DIM']
        DEC_HID_DIM = kwargs['DEC_HID_DIM']
        ENC_DROPOUT = kwargs['ENC_DROPOUT']
        DEC_DROPOUT = kwargs['DEC_DROPOUT']
        device = kwargs['device']
    
        attn = _Attention(ENC_HID_DIM, DEC_HID_DIM)
        enc = _Encoder(ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
        dec = _Decoder(output_dim=OUTPUT_DIM,  enc_hid_dim=ENC_HID_DIM,
                       dec_hid_dim=DEC_HID_DIM, dropout=DEC_DROPOUT, attention=attn)
        self.model = model(enc, dec, device).to(device)
    
        # initialize loss and optimizer
        self.optimiser = optimiser(self.model.parameters(), lr=self.grid['learning_rate'])
        self.loss_function = loss_function().to(device)
    
    def train(self, X_train, y_train, X_val, y_val):
        """
        :param X_train:
            type: numpy.array
            description:
        :param y_train:
            type: numpy.array
            description:
        :param X_val:
            type: numpy.array 
            description:
        :param y_val:
            type: numpy.array
            description:
        """
        ### generate batches
        # training data
        (input_train, input_train_lengths,
         target_train, target_train_lengths) = self._generate_batches(X_train, y_train)
        # validation data
        (input_val, input_val_lengths,
         target_val, target_val_lengths) = self._generate_batches(X_val, y_val)
        
        best_val_loss = float('inf')
    
    @staticmethod
    def _generate_batches(self, input, target):
        """
        :param input:
            type:
            description:
        :param target:
            type:
            description
            
        :return input_batches:
            type:
            description:
        :return input_lengths:
            type:
            description:
        :return target_batches:
            type:
            description:
        :return target_lengths:
            type:
            description:
        """
        %%time
        # determine a number of batches
        n_batches = input.shape[0] // self.batch_size
        
        
        # transform data to the padded array
            # inputs are represented in embedded matrices
            # targets are represented by sequence of corresponding indices
        (padded_input, 
         input_lengths,
         padded_target,
         target_lengths) = self._data2Paddedarray(input, target)
        
        # Generate input and target batches
            #dimension => [total_batchs, seq_length, batch_size, embed_dim], for target embed_dim is irrelevant
        input_batches = np.array(
            np.split(padded_input[:, (n_batches * self.batch_size):, :], n_batches, axis = 1)
            )
        target_batches = np.array(
            np.split(padded_target[:, (n_batches * self.batch_size):], n_batches, axis = 1)
            )
        # Split input and target lenghts into batches as well
        input_lenghts = np.array(
            np.split(input_lengths[(n_batches * self.batch_size):], n_batches, axis = 0)
            )
        target_lenghts = np.array(
            np.split(target_lengths[(n_batches * self.batch_size):], n_batches, axis = 0)
            )
        
        return (input_batches, input_lengths,
                target_batches, target_lenghts)
        
        
    @staticmethod    
    def _data2PaddedArray(self, input, target):
        """
        :param input:
            type:
            description:
        :param target:
            type:
            description
            
        :return embedded_matrix:
            type: numpy.array
            description:
        :return input_seq_lengths:
            type: numpy.array
            description:
        :return padded_target:
            type: numpy.array
            description:
        :return target_seq_lengths:
            type: numpy.array
            description
        """
        # Create a vector of integers representing our text
        numericalVec_input = np.array(
            [[self.text_dictionary.word2index[word] for word in sentence] for sentence in input]
            )
        numericalVec_target = np.array(
            [[self.text_dictionary.word2index[word] for word in sentence] for sentence in target]
            )
        
        ### Convert the input data to embedded representation
        max_lengths = np.array([len(sentence) for sentence in input]).max()
        embedded_matrix, input_seq_lengths = [], []
        for sentence in numericalVec_input:
            # embedding
            embedded_sentence = np.array(
                [self.embeddings[self.text_dictionary.word2index[word]] for word in sentence]
                )
            # append sequence length
            input_seq_lengths.append(
                sentence.shape[0]
                )
            # padding
            if sentence.shape[0] < max_lentghts:
                embedded_sentence = np.r_[embedded_sentence, np.zeros((max_lengths - embedded_sentence.shape[0], self.embeddings.shape[1]))]
            # append embedded sentence
            embedded_matrix.append(embedded_sentence)
        
        ### Pad the target data
        max_lengths = np.array([len(sentence) for sentence in target]).max()
        padded_target, target_seq_lengths = [], []
        for sentence in numericalVec_target:
            if sentence.shape[0] < max_lentghts:
                sentence = np._r[np.array(sentence)]
            else:
                sentence = np.array(sentence, np.zeros((max_lengths - len(sentence),)))
            paddet_target, target_seq_lengths = sentence, sentence.shape[0]
        
        return (np.array(embedded_matrix).float().swapaxes(0,1), # => dims: [seq_length, n_examples, embedded_dim]
                np.array(seq_lengths),
                np.array(padded_target).long().swapaxes(0,1), # => dims: [seq_length, n_examples, embedded_dim]
                np.array(target_seq_lengths)
                )
                
            