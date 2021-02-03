import torch.nn as nn

from radie.src.torchhelper import zeros


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim,
                 batch_size, drop_out_rate=0.5):
        super(BiLSTM, self).__init__()
        self.embedding_dim = embedding_dim

        self.drop_out_rate = drop_out_rate

        self.hidden_dim = hidden_dim
        self.drop_out_rate = drop_out_rate
        self.batch_size = batch_size

        self.dropout = nn.Dropout(self.drop_out_rate)

        self.vocab_size = vocab_size

        self.word_embeds = nn.Embedding(self.vocab_size, self.embedding_dim,
                                        padding_idx=1)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True, batch_first=True)

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (zeros(2, self.batch_size, self.hidden_dim // 2),
                zeros(2, self.batch_size, self.hidden_dim // 2))

    def _get_lstm_features(self, inputs):
        """
        inputs : (batch, seq_length)
        """
        # Initialise hidden state
        self.hidden = self.init_hidden()
        
        embeds = self.word_embeds(inputs)

        if self.drop_out_rate:
            embeds = self.dropout(embeds)

        lstm_out, self.hidden = self.lstm(embeds, self.hidden)

        # lstm_out, _ = self.attention(lstm_out, lstm_out)

        if self.drop_out_rate:
            lstm_out = self.dropout(lstm_out)

        return lstm_out

    def forward(self, inputs):
        """
        inputs : (batch, seq_length)
        """
        self.batch_size = inputs.size(0)
        # Get token representation from the BiLSTM
        outputs = self._get_lstm_features(inputs)

        return outputs