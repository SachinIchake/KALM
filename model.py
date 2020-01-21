import torch
import torch.nn as nn

from embed_regularize import embedded_dropout
from locked_dropout import LockedDropout
from weight_drop import WeightDrop


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, dropouth=0.5, dropouti=0.5, dropoute=0.1,
                 wdrop=0, tie_weights=False,bptt=70):
        super(RNNModel, self).__init__()
        self.lockdrop = LockedDropout()
        self.idrop = nn.Dropout(dropouti)
        self.hdrop = nn.Dropout(dropouth)
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninpExtd = ninp+100
        assert rnn_type in ['LSTM', 'QRNN', 'GRU'], 'RNN type is not supported'
        if rnn_type == 'LSTM':
            self.rnns = [
                torch.nn.LSTM(self.ninpExtd if l == 0 else nhid, nhid if l != nlayers - 1 else (self.ninpExtd if tie_weights else nhid),
                              1, dropout=0) for l in range(nlayers)]
            if wdrop:
                self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop) for rnn in self.rnns]
        # if rnn_type == 'GRU':
        #     self.rnns = [torch.nn.GRU(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else ninp, 1, dropout=0) for l
        #                  in range(nlayers)]
        #     if wdrop:
        #         self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop) for rnn in self.rnns]
        # elif rnn_type == 'QRNN':
        #     from torchqrnn import QRNNLayer
        #     self.rnns = [QRNNLayer(input_size=ninp if l == 0 else nhid,
        #                            hidden_size=nhid if l != nlayers - 1 else (ninp if tie_weights else nhid),
        #                            save_prev_x=True, zoneout=0, window=2 if l == 0 else 1, output_gate=True) for l in
        #                  range(nlayers)]
        #     for rnn in self.rnns:
        #         rnn.linear = WeightDrop(rnn.linear, ['weight'], dropout=wdrop)
        print(self.rnns)
        # self.extSize=500
        self.rnns = torch.nn.ModuleList(self.rnns)
        self.decoder = nn.Linear(self.ninpExtd, ntoken)
        self.w_h = nn.Linear(self.ninpExtd, 100, bias=False)
        # self.vt_1 = torch.zeros(60, 80, 100)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        # if tie_weights:
        #     # if nhid != ninp:
        #     #    raise ValueError('When using the tied flag, nhid must be equal to emsize')
        #     self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.ninp = ninp
        self.nhid = nhid
        self.nlayers = nlayers
        self.dropout = dropout
        self.dropouti = dropouti
        self.dropouth = dropouth
        self.dropoute = dropoute
        self.tie_weights = tie_weights
        self.bptt=bptt

    def reset(self):
        if self.rnn_type == 'QRNN': [r.reset() for r in self.rnns]

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, w_e,vt_1, return_h=False):
        emb = embedded_dropout(self.encoder, input, dropout=self.dropoute if self.training else 0)
        # emb = self.idrop(emb)

        emb = self.lockdrop(emb, self.dropouti)
        # emb = torch.cat((emb, self.vt_1), len(emb.size()) - 1)

        print('emb: ' , emb.size())
        print('vt_1: ', vt_1.size())
        emb = torch.cat((emb, vt_1[:emb.size(0)]), len(emb.size()) - 1)
        # emb = torch.cat((emb, vt_1), len(emb.size()) - 1)

        raw_output = emb
        new_hidden = []
        # raw_output, hidden = self.rnn(emb, hidden)
        raw_outputs = []
        outputs = []
        for l, rnn in enumerate(self.rnns):
            current_input = raw_output
            raw_output, new_h = rnn(raw_output, hidden[l])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if l != self.nlayers - 1:
                # self.hdrop(raw_output)
                raw_output = self.lockdrop(raw_output, self.dropouth)
                outputs.append(raw_output)
        hidden = new_hidden

        output = self.lockdrop(raw_output, self.dropout)
        outputs.append(output)

        '''
        KALM Start
        '''
        # Extended the output for 5 different types of vocab
        output_xp = output.unsqueeze(dim=2)
        output_xp = output_xp.repeat(1, 1, 5, 1)

        # Decoded the output
        decoded = self.decoder(
            output_xp.view(output_xp.size(0) * output_xp.size(1) * output_xp.size(2), output_xp.size(3)))
        resultRNN = decoded.view(output_xp.size(0), output_xp.size(1), output_xp.size(2), decoded.size(1))
        resultSoftmax = nn.Softmax(resultRNN)

        ''' 
        Type Embedding
        '''
        LowerHt = self.w_h(output.view(output.size(0) * output.size(1), output.size(2)))
        typeCalc = torch.matmul(LowerHt, torch.t(w_e))
        typeCalc = typeCalc.view(output.size(0), output.size(1), typeCalc.size(1))
        pieT_1 = nn.Softmax(typeCalc)
        result = pieT_1.dim.data[:, :, :, None]
        resultMulSoftmax = torch.mul(resultSoftmax.dim.data, result)

        result = resultMulSoftmax.sum(dim=2)
        # result = result_sum.argmax(dim=2)
        vt_1 = torch.matmul(pieT_1.dim.data, w_e)

        '''
        KALM End
        '''

        result = output.view(output.size(0) * output.size(1), output.size(2))
        if return_h:
            return result, hidden, raw_outputs, outputs,vt_1
        return result, hidden,vt_1

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return [(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (
                self.ninpExtd if self.tie_weights else self.nhid)).zero_(),
                     weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (
                         self.ninpExtd if self.tie_weights else self.nhid)).zero_())
                    for l in range(self.nlayers)], (weight.new(5, 100)),(weight.new(self.bptt,bsz, 100))
        elif self.rnn_type == 'QRNN' or self.rnn_type == 'GRU':
            return [weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (
                self.ninp if self.tie_weights else self.nhid)).zero_()
                    for l in range(self.nlayers)]
