import math

import torch.nn as nn
import torch
import numpy as np
from vocab import Vocab


# class EncoderAttention(nn.Module):
#     def __init__(self, vocab_size, embed_size, hidden_size,
#                  n_layers=1, dropout=0.5):
#         super(EncoderAttention, self).__init__()
#         self.vocab_size = vocab_size
#         self.hidden_size = hidden_size
#         self.embed_size = embed_size
#         self.n_layers = n_layers
#
#         self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)
#         self.gru = nn.GRU(embed_size, hidden_size, n_layers,
#                           dropout=dropout, bidirectional=True)
#
#         self.dropout = nn.Dropout(dropout)
#
#
#     def forward(self, src, hidden=None):
#         embedded = self.embed(src)
#         outputs, hidden = self.gru(embedded, hidden)
#         # sum bidirectional outputs
#         outputs = (outputs[:, :, :self.hidden_size] +
#                    outputs[:, :, self.hidden_size:])
#         return outputs, hidden


# class Attention(nn.Module):
#     def __init__(self, hidden_size):
#         super(Attention, self).__init__()
#         self.hidden_size = hidden_size
#         self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
#         self.v = nn.Parameter(torch.rand(hidden_size))
#         self.tanh = nn.Tanh()
#         self.Wa = nn.Linear(hidden_size * 2, hidden_size, bias=False)
#         self.Ua = nn.Linear(hidden_size * 2, hidden_size, bias=False)
#         # self.va = nn.Parameter(torch.FloatTensor(batch_size, hidden_size))
#         #
#         self.softmax = nn.Softmax(dim=1)
#         stdv = 1. / math.sqrt(self.v.size(0)) #init like the paper
#         self.v.data.uniform_(-stdv, stdv)
#
#     def forward(self, hidden, encoder_outputs):
#         timestep = encoder_outputs.size(0)
#         h = hidden.repeat(timestep, 1, 1).transpose(0, 1)
#         encoder_outputs = encoder_outputs.transpose(0, 1)  # [B*T*H]
#         attn_energies = self.score(h, encoder_outputs)
#         return self.softmax(attn_energies).unsqueeze(1)
#
#     # def score(self,):
#     #     x = last_hidden.unsqueeze(1)
#     #     out = self.tanh(self.Wa(x) + self.Ua(encoder_outputs))
#     #     return out.bmm(self.va.unsqueeze(2)).squeeze(-1)
#
#
#
#     def score(self, hidden, encoder_outputs):
#             # [B*T*2H]->[B*T*H]
#             energy = self.relu(self.attn(torch.cat([hidden, encoder_outputs], 2)))
#             energy = energy.transpose(1, 2)  # [B*H*T]
#             v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # [B*1*H]
#             energy = torch.bmm(v, energy)  # [B*1*T]
#             return energy.squeeze(1)  # [B*T]


class Attention(nn.Module):
    # """ Applies attention mechanism on the `context` using the `query`.
    #
    # **Thank you** to IBM for their initial implementation of :class:`Attention`. Here is
    # their `License
    # <https://github.com/IBM/pytorch-seq2seq/blob/master/LICENSE>`__.
    #
    # Args:
    #     dimensions (int): Dimensionality of the query and context.
    #     attention_type (str, optional): How to compute the attention score:
    #
    #         * dot: :math:`score(H_j,q) = H_j^T q`
    #         * general: :math:`score(H_j, q) = H_j^T W_a q`
    #
    # Example:
    #
    #      attention = Attention(256)
    #      query = torch.randn(5, 1, 256)
    #      context = torch.randn(5, 5, 256)
    #      output, weights = attention(query, context)
    #      output.size()
    #      torch.Size([5, 1, 256])
    #      weights.size()
    #      torch.Size([5, 1, 5])
    # """

    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.concat_linear = nn.Linear(self.hidden_dim * 2 * 2, self.hidden_dim)
        self.attn = nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2)
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()

    def forward(self, rnn_outputs, final_hidden_state):
        # rnn_output.shape:         (seq_len, batch_size , hidden_size)
        # final_hidden_state.shape: (batch_size, hidden_size)
        # NOTE: hidden_size may also reflect bidirectional hidden states (hidden_size = num_directions * hidden_dim)
        rnn_outputs = rnn_outputs.transpose(0, 1)
        batch_size, seq_len, _ = rnn_outputs.shape
        attn_weights = self.attn(rnn_outputs) # (batch_size, seq_len, hidden_dim)
        attn_weights = self.tanh(attn_weights)
        attn_weights = torch.bmm(attn_weights, final_hidden_state.unsqueeze(2))
        attn_weights = self.softmax(attn_weights.squeeze(2))

        return attn_weights


class DecoderAttention(nn.Module):
    def __init__(self, hidden_size, vocab_size,embed_size,
                 n_layers=1, dropout=0.2):
        super(DecoderAttention, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.num_directions = 2
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.attention = Attention(hidden_size)
        self.lstm = nn.LSTM(embed_size + hidden_size * 2, hidden_size, n_layers,bidirectional=True,
                            dropout=dropout)
        self.out_linear = nn.Linear(hidden_size * 4, vocab_size)

    def forward(self, input_tok, hidden, cell, encoder_outputs):
        batch_size = input_tok.size()[0]
        # Get the embedding of the current input word (last output word)
        embedded = self.embedding(input_tok)
        embedded = self.dropout(embedded)
        # Calculate attention weights and apply to encoder outputs

        #Get the hidden size from the layers and directions
        final_state = hidden.view(self.n_layers, self.num_directions, batch_size, self.hidden_size)[-1]

        #Because num of directions is always 2 now we need to ge both directions
        h_1, h_2 = final_state[0], final_state[1]
        # final_hidden_state = h_1 + h_2  -> Add both states (requires changes to the input size of first linear layer + attention layer)
        final_hidden_state = torch.cat((h_1, h_2), 1)  # Concatenate both states

        attn_weights = self.attention(encoder_outputs, final_hidden_state)
        context = torch.bmm(encoder_outputs.transpose(0, 1).transpose(1, 2), attn_weights.unsqueeze(2)).squeeze(2)
        # attn_hidden = torch.cat((context, final_hidden_state), dim=1)
        # Combine embedded input word and attended context, run through RNN
        rnn_input = torch.cat([embedded, context], 1)
        output,  (hidden, cell) = self.lstm(rnn_input.unsqueeze(0),  (hidden, cell))
        output = output.squeeze(0)  # (1,B,N) -> (B,N)
        context = context.squeeze(0) if len(context.shape) == 3 else context #dealing with batch size 1
        output = self.out_linear(torch.cat([output, context], 1))
        return output, hidden, attn_weights


class EncoderVanilla(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size,
                 n_layers=1, dropout=0.2):
        super(EncoderVanilla, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.n_layers = n_layers
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(embed_size, hidden_size, n_layers,bidirectional=True,
                            dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, source):
        embedded = self.embedding(source)
        embedded = self.dropout(embedded)
        embedded = embedded.transpose(0, 1) # make it (seq_len, batch_size, features)
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, hidden, cell



class DecoderVanilla(nn.Module):
    def __init__(self, hidden_size, vocab_size, embed_size,
                 n_layers=1, dropout=0.5):
        super(DecoderVanilla, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embed_size = embed_size

        self.n_layers = n_layers
        self.embedding = nn.Embedding(self.vocab_size, embed_size * 2, padding_idx=0)
        self.lstm = nn.LSTM(embed_size * 2, hidden_size, n_layers,bidirectional=True,
                          dropout=dropout)

        self.out_linear = nn.Linear(hidden_size*2, self.vocab_size)
        self.relu = nn.ReLU()

    def forward(self, input_tok, hidden, cell):
        # input is the previous predicted word
        output = self.embedding(input_tok)
        output = self.relu(output)
        output, (hidden, cell) = self.lstm(output.unsqueeze(0), (hidden, cell))
        output = self.out_linear(output) #  N * 1 * lenght_of_vocab
        output = output.squeeze(0)
        return output, (hidden, cell)


class Seq2Seq(nn.Module):
    max_len = 50
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.encoder = encoder
        self.decoder = decoder
        assert encoder.hidden_size == decoder.hidden_size, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"


    def forward(self, src, trg, train):
        batch_size = src.size(0)
        max_len = trg.size(1) if train else self.max_len
        vocab_size = self.decoder.vocab_size
        #init the strat output with 1 in the start index and 0 on the rest
        start_output = np.zeros((batch_size, vocab_size))
        start_output[:, 1] = 1
        outputs = [torch.tensor(start_output,  device=self.device, dtype=torch.float64, requires_grad=True)]
        encoder_output, hidden, cell = self.encoder(src)
        output = trg[:, 0] if trg is not None else torch.ones(batch_size, device=self.device).type(torch.int) #init start token as an array - Encoder forword will transpose them
        for t in range(1, max_len):
            if type(self.decoder) == DecoderVanilla:
                output, (hidden, cell) = self.decoder(
                    output, hidden, cell)
            else: #attention
                output, hidden, attn_weights = self.decoder(
                    output, hidden,cell, encoder_output)
            outputs.append(output)
            if train:
                output = trg[:, t]
            else:
                output[:, 0] = -100  # Dont allow prediction of Padding index.
                output = output.argmax(1)
                if all([i == Vocab.END_IDX for i in output.cpu().numpy()]): # Support Dev only for batch in size of 1
                    break

        outputs = torch.stack(outputs, dim=1)
        return outputs
