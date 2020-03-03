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

# ----- Settings -----
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

exec(open('Code/Models/Attention_seq2seq.py').read())
# ----- Settings -----

class generator:
    """
    """
    def __init__(self, model, loss_function,
                 optimiser, batch_size, 
                 text_dictionary, embeddings, 
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
         target_train, target_train_lengths) = self._generate_batches(input = X_train,
                                                                      target = y_train)
        # validation data
        (input_val, input_val_lengths,
         target_val, target_val_lengths) = self._generate_batches(input = X_val,
                                                                  target = y_val)
        
        # Initialize empty lists for training and validation loss + put best_val_loss = +infinity
        self.train_losses, self.val_losses = [], []
        self.best_val_loss = float('inf')
        # run the training
        self.model.train()
        for epoch in range(self.grid['max_epochs']):
            epoch_loss = 0
            
            for input, target, seq_length_input, seq_length_target in zip(input_train,
                                                                          target_train,
                                                                          input_train_lengths,
                                                                          target_train_lengths
                                                                          ):
                # zero gradient
                self.optimiser.zero_grad()
                ## FORWARD PASS
                # Prepare RNN-edible input - i.e. pack padded sequence
                input = nn.utils.rnn.pack_padded_sequence(torch.from_numpy(input).float(),
                                                          lengths = seq_length_input,
                                                          batch_first = False,
                                                          enforce_sorted = False).to(device)
                output = self.model(seq2seq_input = input, target = target,
                                    teacher_forcing_ratio = self.teacher_forcing_ratio
                                    )
                del input
                # Pack output and target padded sequence
                ## Determine a length of output sequence based on the first occurrence of <eos>
                seq_length_output = np.array(
                    [out == self.text_dictionary.word2index['<eos>'] for out in output.transpose()]
                    ).argmax(1)
                seq_length_output = np.array(
                    [seq_length_output.shape[0] if seq_len == 0 else seq_len for seq_len in seq_length_input]
                    )
                # determine seq_length for computation of loss function based on max(seq_lenth_target, seq_length_output)
                seq_length_loss = np.array(
                    (seq_length_output, seq_length_target)
                    ).max(0)
                
                output = nn.utils.rnn.pack_padded_sequence(output,
                                                           lengths = seq_length_loss,
                                                           batch_first = False,
                                                           enforce_sorted = False).to(device)
                
                target = nn.utils.rnn.pack_padded_sequence(torch.from_numpy(target).long(),
                                                           lengths = seq_length_loss,
                                                           batch_first = False,
                                                           enforce_sorted = False).to(device)
                
                # Compute loss
                loss = self.loss_function(output[0], target[0])
                del output, target
                
                ### BACKWARD PASS
                # Make update step w.r.t. clipping gradient
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grid['clip'])
                self.optimiser.step()
                
                self.epoch_loss += loss.item()
           
            # Save training loss and validation loss
            self.train_losses.append(epoch_loss)
            self.val_losses.append(
                self._evaluate(input_val, input_val_lengths,
                               target_val, target_val_lengths)
                )
            
            # Store the best model if validation loss improved
            if self.val_losses[epoch] < self.best_val_loss:
                self.best_val_loss = self.val_losses[epoch]
                self.m = copy.deepcopy(self.model)
            
            # Print the progress
            print(f'Epoch: {epoch+1}:')
            print(f'Train Loss: {self.train_losses[epoch]:.3f}')
            print(f'Validation Loss: {self.val_losses[epoch]:.3f}')
            
                

    def _evaluate(self, input_val, input_val_lengths, target_val, target_val_lengths):
        """
        :param input_val:
            type:
            description:
        :param input_val_lengths:
            type:
            description:
        :param target_val:
            type:
            description:
        :param target_val_lengths:
            type:
            description:
                
        :return val_loss:
            type:
            description:
        """
        self.model.eval()
        val_loss = 0
        for input, target, seq_length_input, seq_length_target in zip(input_val,
                                                                      target_val,
                                                                      input_val_lengths,
                                                                      target_val_lengths
                                                                      ):
            input = nn.utils.rnn.pack_padded_sequence(torch.from_numpy(input).float(),
                                                      lengths = seq_length_input,
                                                      batch_first = False,
                                                      enforce_sorted = False).to(device)
            output = self.model(seq2seq_input = input, target = target,
                                teacher_forcing_ratio = self.teacher_forcing_ratio
                                )
            del input
            # Pack output and target padded sequence
            ## Determine a length of output sequence based on the first occurrence of <eos>
            seq_length_output = np.array(
                [out == self.text_dictionary.word2index['<eos>'] for out in output.transpose()]
                ).argmax(1)
            seq_length_output = np.array(
                [seq_length_output.shape[0] if seq_len == 0 else seq_len for seq_len in seq_length_input]
                )
            # determine seq_length for computation of loss function based on max(seq_lenth_target, seq_length_output)
            seq_length_loss = np.array(
                (seq_length_output, seq_length_target)
                ).max(0)
            
            output = nn.utils.rnn.pack_padded_sequence(output,
                                                       lengths = seq_length_loss,
                                                       batch_first = False,
                                                       enforce_sorted = False).to(device)
            
            target = nn.utils.rnn.pack_padded_sequence(torch.from_numpy(target).long(),
                                                       lengths = seq_length_loss,
                                                       batch_first = False,
                                                       enforce_sorted = False).to(device)
            
            # Compute loss
            val_loss += self.loss_function(output[0], target[0]).item()
        
        return val_loss
    
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
        # determine a number of batches
        n_batches = input.shape[0] // self.batch_size
        
        
        # transform data to the padded array
            # inputs are represented in embedded matrices
            # targets are represented by sequence of corresponding indices
        (padded_input, 
         input_lengths,
         padded_target,
         target_lengths) = self._data2PaddedArray(input, target)
        
        # Generate input and target batches
            #dimension => [total_batchs, seq_length, batch_size, embed_dim], for target embed_dim is irrelevant
                #seq_length is variable throughout the batches
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
        
        # trim sequences in individual batches
        for batch in n_batches:
            input_batches[batch] = input_batches[batch, :input_lenghts[batch].max(), :, :]
            target_batches[batch] = target_batches[batch, :target_lenghts[batch].max(), :, :]
        
        # return prepared data
        return (input_batches, input_lengths,
                target_batches, target_lenghts)
               
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
            [[self.__word2index__(word) for word in sentence] for sentence in input]
            )
        numericalVec_target = np.array(
            [[self.__word2index__(word) for word in sentence] for sentence in target]
            )
        
        ### Convert the input data to embedded representation
        max_lengths = np.array([len(sentence) for sentence in input]).max()
        embedded_matrix, input_seq_lengths = [], []
        for sentence in numericalVec_input:
            # embedding
            embedded_sentence = np.array(
                [self.embeddings[self.__word2index__(word)] for word in sentence]
                )
            # append sequence length
            input_seq_lengths.append(
                len(sentence)
                )
            # padding
            if len(sentence) < max_lengths:
                embedded_sentence = np.r_[embedded_sentence, np.zeros((max_lengths - embedded_sentence.shape[0], self.embeddings.shape[1]))]
            # append embedded sentence
            embedded_matrix.append(embedded_sentence)
        
        ### Pad the target data
        max_lengths = np.array([len(sentence) for sentence in target]).max()
        padded_target, target_seq_lengths = [], []
        for sentence in numericalVec_target:
            if len(sentence) < max_lengths:
                sentence = np._r[np.array(sentence)]
            else:
                sentence = np.array(sentence, np.zeros((max_lengths - len(sentence),)))
            paddet_target, target_seq_lengths = sentence, len(sentence)
        
        return (np.array(embedded_matrix).float().swapaxes(0,1), # => dims: [seq_length, n_examples, embedded_dim]
                np.array(seq_lengths),
                np.array(padded_target).long().swapaxes(0,1), # => dims: [seq_length, n_examples, embedded_dim]
                np.array(target_seq_lengths)
                )
    
    def __word2index__(self, word):
        """
        :param word:
            type:
            description:
        """
        try:
            word2index = self.text_dictionary.word2index[word]
        except:
            word2index = self.embeddings.shape[1] - 1
        return word2index
