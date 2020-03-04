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

### Show parameter
def show_parameter():

    print(
        ''' 
    This is a seq2seq model, embedding should be done before input into this model
    
    RNN used is GRU
    
    default loss function is MSELoss()

    run function: instan_things,
         to instantiate your model, 
            in which you should define the following dictionary parameters
    e.g.
    param = {'max_epochs':64,
            'learning_rate':1e-3,       
            'clip':1,                  # clip grad norm
            'teacher_forcing_ratio':1, # during training
            'OUTPUT_DIM':1,            # intented output dimension
            'ENC_EMB_DIM':21,          # embedding space of your input
            'ENC_HID_DIM':32,          
            'DEC_HID_DIM':32,          # hidden dimension should be the same
            'ENC_DROPOUT':0,
            'DEC_DROPOUT':0,
            'device':device}
      
    Training:
    seq2seq_running(grid, model, optimiser, lossfunction, X_train, y_train, X_test, y_test, teacher_forcing_ratio)
    
    Evaluation:
    seq2seq_evaluate(model, X_test, y_test, lossfunction)
    
    Prediction:
    model(self, seq2seq_input, target, teacher_forcing_ratio = 0)
    
    in which:
    seq2seq_input = [seq_len, batch size,Enc_emb_dim]
    target = [trg_len, batch size,output_dim], trg_len is prediction len
    
    '''
    )


### Encoder
class _Encoder(nn.Module):
    """
    """
    def __init__(self, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        """
        :param emb_dim:
            type:
            description:
        :param enc_hid_dim:
            type:
            description:
        :param dec_hid_dim:
            type:
            description:
        :param dropout:
            type:
            description:
        """
        super().__init__()
        self.rnn = nn.GRU(input_size=emb_dim,
                          hidden_size=enc_hid_dim, bidirectional=True)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, enc_input):
        """
        :param enc_input:
            type:
            description:
                
        :return output:
            type:
            description:
        :retun hidden:
            type:
            description:
        """
        # enc_input = [enc_input_len, batch size,emb_dim]

        embedded = enc_input
        #embedded[0] = self.dropout(enc_input[0])  # embedded = [enc_input_len, batch size, emb_dim]

        outputs, hidden = self.rnn(embedded)

        # outputs = [enc_input len, batch size, hid dim * num directions]
        # hidden = [n layers * num directions, batch size, hid dim]

        #hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        # outputs are always from the last layer

        # hidden [-2, :, : ] is the last of the forwards RNN
        # hidden [-1, :, : ] is the last of the backwards RNN

        # initial decoder hidden is final hidden state of the forwards and backwards
        #  encoder RNNs fed through a linear layer
        hidden = torch.tanh(
            self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))

        # outputs = [src len, batch size, enc hid dim * 2]
        # hidden = [batch size, dec hid dim]
        return outputs, hidden     
        
### Attention
class _Attention(nn.Module):
    """
    """
    def __init__(self, enc_hid_dim, dec_hid_dim):
        """
        :param enc_hid_dim:
            type:
            description:
        :param dec_hid_dim:
            type:
            description:
        """
        super().__init__()

        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        """
        :param hidden:
            type:
            description:
        :param encoder_outputs:
            type:
            description
        
        :return softmax(attention):
            type:
            description:
        """

        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [enc_seq_len, batch size, enc hid dim * 2]

        batch_size = encoder_outputs[0].shape[1]
        enc_seq_len = encoder_outputs[0].shape[0]

        # repeat decoder hidden state enc_seq_len times
        hidden = hidden.unsqueeze(1).repeat(1, enc_seq_len, 1)
        print(encoder_outputs[0].shape)
        encoder_outputs = encoder_outputs[0].permute(1, 0, 2)

        # hidden = [batch size, enc_seq_len, dec hid dim]
        # encoder_outputs = [batch size, enc_seq_len, enc hid dim * 2]

        energy = torch.tanh(
            self.attn(torch.cat((hidden, encoder_outputs), dim=2))) # energy = [batch size, enc_seq_len, dec hid dim]

        attention = self.v(energy).squeeze(2)   # attention= [batch size, enc_seq_len]

        return F.softmax(attention, dim=1)
    
class _Decoder(nn.Module):
    """
    """
    def __init__(self, output_dim, enc_hid_dim,  dec_hid_dim, dropout, attention):
        """
        :param output_dim:
            type:
            description:
        :param enc_hid_dim:
            type:
            description:
        :param dec_hid_dim:
            type:
            description:
        :param dropout:
            type:
            description:
        :param attention:
            type:
            description:
        """
        super().__init__()

        self.output_dim = output_dim
        self.attention = attention

        #self.embedding = nn.Embedding(output_dim, output_dim)

        self.rnn = nn.GRU((enc_hid_dim * 2) + output_dim, dec_hid_dim)

        self.fc_out = nn.Linear(
            (enc_hid_dim * 2) + dec_hid_dim + output_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, dec_input, hidden, encoder_outputs):
        """
        :param dec_input:
            type:
            decription:
        :param hidden:
            type:
            description:
        :param encoder_outputs:
            type:
            description:
                
        :return prediction:
            type:
            description:
        :return hidden:
            type:
            description:
        """

        # dec_input = [1,batch size,dec_emb dim]
        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [enc_seq_len, batch size, enc hid dim * 2]

        #embedded = self.dropout(dec_input)  # embedded = [1, batch size, dec_emb dim]

        attention = self.attention(hidden, encoder_outputs) # attention = [batch size, enc_seq_len]

        attention = attention.unsqueeze(1)  # attention = [batch size, 1, enc_seq_len]

        encoder_outputs = encoder_outputs.permute(1, 0, 2)  # encoder_outputs = [batch size, enc_seq_len, enc hid dim * 2]

        weighted = torch.bmm(attention, encoder_outputs)    # weighted = [batch size, 1, enc hid dim * 2]

        weighted = weighted.permute(1, 0, 2) # weighted = [1, batch size, enc hid dim * 2]

        # print('embedded',embedded.size())
        rnn_input = torch.cat((embedded, weighted), dim=2)  # rnn_input = [1, batch size, (enc hid dim * 2) + dec_emb dim]

        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))

        # output = [seq len, batch size, dec hid dim * n directions]
        # hidden = [n layers * n directions, batch size, dec hid dim]

        # seq len, n layers and n directions will always be 1 in this decoder, therefore:
        # output = [1, batch size, dec hid dim]
        # hidden = [1, batch size, dec hid dim]
        # this also means that output == hidden
        assert (output == hidden).all()

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)

        prediction = self.fc_out(
            torch.cat((output, weighted, embedded), dim=1))

        # prediction = [batch size, output dim]

        return prediction, hidden.squeeze(0)
    
class _Seq2Seq(nn.Module):
    """
    """
    def __init__(self, encoder, decoder, device):
        """
        :param encoder:
            type:
            description:
        :param decoder:
            type:
            description:
        :param device:
            type:
            description
        """
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, seq2seq_input, target, teacher_forcing_ratio=0.5):
        """
        :param seq2seq_input:
            type:
            description:
        :param target:
            type:
            description:
        :param teacher_forcing_ratio:
            type:
            description:
                
        :return outputs:
            type:
            description:
        """
        # seq2seq_input = [seq_len, batch size,Enc_emb_dim]
        # target = [trg_len, batch size,output_dim]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of the time

        batch_size = seq2seq_input[0].shape[1]
        trg_len = target[0].shape[0]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size,
                              trg_vocab_size).to(self.device)

        # encoder_outputs is all hidden states of the input sequence, back and forwards
        # hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_outputs, hidden = self.encoder(seq2seq_input)
        
        # check: make dimension consistent
        dec_input = target[0][0]
        dec_input = dec_input.unsqueeze(0)
        # print('dec_input dim:',dec_input.size())

        for t in range(1, trg_len):
            # insert dec_input token embedding, previous hidden state and all encoder hidden states
            # receive output tensor (predictions) and new hidden state
            output, hidden = self.decoder(dec_input, hidden, encoder_outputs)

            # place predictions in a tensor holding predictions for each token
            outputs[t] = output

            # decide if we are going to use teacher forcing or not
            teacher_force = np.random.random() < teacher_forcing_ratio

            # get the highest predicted token from our predictions
            top1 = output.argmax(1)
            top1 = top1.unsqueeze(1)

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            dec_input = target[t] if teacher_force else top1
            dec_input = dec_input.unsqueeze(1).float()
        return outputs

    def save(self, name_path):
        """
        :param name_path:
            type:
            description:
        """
        torch.save(self.state_dict(), name_path)  # e.g. 'encoder_model.pt'

    def load(self, name_path):
        """
        :param name_path:
            type:
            description:
        """
        self.load_state_dict(torch.load(name_path))
        self.eval()
