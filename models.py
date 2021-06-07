import torch.nn as nn
import torch

class EncoderAttention(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size,
                 n_layers=1, dropout=0.5):
        super(EncoderAttention, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, n_layers,
                          dropout=dropout, bidirectional=True)

    def forward(self, src, hidden=None):
        embedded = self.embed(src)
        outputs, hidden = self.gru(embedded, hidden)
        # sum bidirectional outputs
        outputs = (outputs[:, :, :self.hidden_size] +
                   outputs[:, :, self.hidden_size:])
        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        torch.nn.init.xavier_uniform_(self.v)

    def forward(self, hidden, encoder_outputs):
        timestep = encoder_outputs.size(0)
        h = hidden.repeat(timestep, 1, 1).transpose(0, 1)
        encoder_outputs = encoder_outputs.transpose(0, 1)  # [B*T*H]
        attn_energies = self.score(h, encoder_outputs)
        return self.softmax(attn_energies, dim=1).unsqueeze(1)

    def score(self, hidden, encoder_outputs):
        # [B*T*2H]->[B*T*H]
        energy = self.relu(self.attn(torch.cat([hidden, encoder_outputs], 2)))
        energy = energy.transpose(1, 2)  # [B*H*T]
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # [B*1*H]
        energy = torch.bmm(v, energy)  # [B*1*T]
        return energy.squeeze(1)  # [B*T]


class DecoderAttention(nn.Module):
    def __init__(self, embed_size, hidden_size, output_size,
                 n_layers=1, dropout=0.2):
        super(DecoderAttention, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.embed = nn.Embedding(output_size, embed_size)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.attention = Attention(hidden_size)
        self.gru = nn.GRU(hidden_size + embed_size, hidden_size,
                          n_layers, dropout=dropout)
        self.out = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input, last_hidden, encoder_outputs):
        # Get the embedding of the current input word (last output word)
        embedded = self.embed(input).unsqueeze(0)  # (1,B,N)
        embedded = self.dropout(embedded)
        # Calculate attention weights and apply to encoder outputs
        attn_weights = self.attention(last_hidden[-1], encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # (B,1,N)
        context = context.transpose(0, 1)  # (1,B,N)
        # Combine embedded input word and attended context, run through RNN
        rnn_input = torch.cat([embedded, context], 2)
        output, hidden = self.gru(rnn_input, last_hidden)
        output = output.squeeze(0)  # (1,B,N) -> (B,N)
        context = context.squeeze(0)
        output = self.out(torch.cat([output, context], 1))
        return output, hidden, attn_weights


class EncoderVanilla(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size,
                 n_layers=1, dropout=0.2):
        super(EncoderVanilla, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(embed_size, hidden_size, n_layers,bidirectional=True,
                            dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, source, hidden=None):
        #src = [src len, batch size]
        embedded = self.embedding(source)
        embedded = self.dropout(embedded)

        #embedded = [src len, batch size, emb dim]

        outputs, (hidden, cell) = self.lstm(embedded)

        #outputs = [src len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]

        #outputs are always from the top hidden layer

        return hidden, cell



class DecoderVanilla(nn.Module):
    def __init__(self, hidden_size, vocab_size, embed_size,
                 n_layers=1, dropout=0.5):
        super(DecoderVanilla, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.embedding = nn.Embedding(self.vocab_size, embed_size * 2, padding_idx=0)
        self.lstm = nn.LSTM(embed_size * 2, hidden_size, n_layers,bidirectional=True,
                          dropout=dropout, batch_first=True)

        self.out_linear = nn.Linear(hidden_size*2, self.vocab_size)
        self.relu = nn.ReLU()

    def forward(self, input_tok, hidden, cell):
        # input is the previous predicted word
        output = self.embedding(input_tok)
        output = self.relu(output)
        output, (hidden, cell) = self.lstm(output.unsqueeze(1), (hidden, cell))
        #output = [seq len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]

        #seq len and n directions will always be 1 in the decoder, therefore:
        #output = [1, batch size, hid dim]
        #hidden = [n layers, batch size, hid dim]
        #cell = [n layers, batch size, hid dim]
        #Maybe do squeeze(1)
        output = self.out_linear(output) #N * 1 * lenght_of_vocab
        output = output.squeeze(1)
        return output, (hidden, cell)






class Seq2Seq(nn.Module):
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
        max_len = trg.size(1)
        vocab_size = self.decoder.vocab_size
        outputs = torch.zeros(batch_size, 1, vocab_size, requires_grad=True, device=self.device)

        hidden, cell = self.encoder(src)

        output = trg[:, 0]
        for t in range(1, max_len):
            if type(self.encoder) == EncoderVanilla:
                output, (hidden, cell) = self.decoder(
                    output, hidden, cell)
            else:
                pass
                # output, hidden, attn_weights = self.decoder(
                #     output, hidden, encoder_output)
            outputs = torch.cat((outputs, output.unsqueeze(0)),dim=1)
            if train:
                output = trg[:, t]
            else:
                output[:, 0] = -100 # Dont allow prediction of Padding index.
                output = output.argmax(1)
        return outputs
