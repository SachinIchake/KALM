import torch
import torch.nn as nn
import numpy as np
from embed_regularize import embedded_dropout
from locked_dropout import LockedDropout
from weight_drop import WeightDrop
from torch.autograd import Variable


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, vOrgLen, vLocLen, vPersonLen, vGeneralLen, vMiscLen, ninp, nhid, nlayers,
                 dropout=0.5, dropouth=0.5, dropouti=0.5, dropoute=0.1, wdrop=0, ntypes=4, ntypesDims=100,
                 tie_weights=False):
        super(RNNModel, self).__init__()
        self.lockdrop = LockedDropout()
        self.idrop = nn.Dropout(dropouti)
        self.hdrop = nn.Dropout(dropouth)
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        # self.orgEmb = nn.Embedding(vOrgLen,ntypesDims)
        # self.locEmb = nn.Embedding(vLocLen, ntypesDims)
        # self.perEmb = nn.Embedding(vPersonLen, ntypesDims)
        # self.genEmb = nn.Embedding(vGeneralLen, ntypesDims)
        # self.mscEmb = nn.Embedding(vMiscLen, ntypesDims)

        assert rnn_type in ['LSTM', 'QRNN', 'GRU'], 'RNN type is not supported'
        if rnn_type == 'LSTM':
            self.rnns = [
                torch.nn.LSTM(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else (ninp if tie_weights else nhid),
                              1, dropout=0) for l in range(nlayers)]
            if wdrop:
                self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop) for rnn in self.rnns]

        print(self.rnns)
        self.rnns = torch.nn.ModuleList(self.rnns)
        self.decoder = nn.Linear(nhid, ntoken)
        self.orgDecoder = nn.Linear(nhid, vOrgLen)
        self.locDecoder = nn.Linear(nhid, vLocLen)
        self.perDecoder = nn.Linear(nhid, vPersonLen)
        self.genDecoder = nn.Linear(nhid, vGeneralLen)
        self.mscDecoder = nn.Linear(nhid, vMiscLen)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            # if nhid != ninp:
            #    raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

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

    def reset(self):
        if self.rnn_type == 'QRNN': [r.reset() for r in self.rnns]

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, vOrg, vLoc, vPerson, vGeneral, vMisc, hidden, return_h=False):
        emb = embedded_dropout(self.encoder, input, dropout=self.dropoute if self.training else 0)
        emb = self.lockdrop(emb, self.dropouti)
        raw_output = emb
        # raw_output = torch.cat((emb, emb_type), len(emb.size()) - 1 + len(emb_type.size())-1)

        # raw_output = emb
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

      # result_1 = output.view(output.size(0)*output.size(1), output.size(2))
        decoded = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))

        orgDecoder = self.orgDecoder(output.view(output.size(0) * output.size(1), output.size(2)))
        locDecoder = self.locDecoder(output.view(output.size(0) * output.size(1), output.size(2)))
        perDecoder = self.perDecoder(output.view(output.size(0) * output.size(1), output.size(2)))
        genDecoder = self.genDecoder(output.view(output.size(0) * output.size(1), output.size(2)))
        mscDecoder = self.mscDecoder(output.view(output.size(0) * output.size(1), output.size(2)))

        resultAll = decoded.view(output.size(0), output.size(1), decoded.size(1))
        resultOrgAll = orgDecoder.view(output.size(0), output.size(1), orgDecoder.size(1))
        resultLocAll = locDecoder.view(output.size(0), output.size(1), locDecoder.size(1))
        resultPerAll = perDecoder.view(output.size(0), output.size(1), perDecoder.size(1))
        resultGenAll = genDecoder.view(output.size(0), output.size(1), genDecoder.size(1))
        resultMScAll = mscDecoder.view(output.size(0), output.size(1), mscDecoder.size(1))

        # OrgSoftmax= nn.Softmax(resultOrgAll)
        # OrgSoftmax = nn.Softmax(resultOrgAll)
        # OrgSoftmax = nn.Softmax(resultOrgAll)
        # OrgSoftmax = nn.Softmax(resultOrgAll)
        # OrgSoftmax = nn.Softmax(resultOrgAll)
        # OrgSoftmax = nn.Softmax(resultOrgAll)


        if return_h:
            return resultAll, resultOrgAll, resultLocAll, resultPerAll, resultGenAll, resultMScAll, hidden, raw_outputs, outputs
        return resultAll, resultOrgAll, resultLocAll, resultPerAll, resultGenAll, resultMScAll, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return [(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (
                self.ninp if self.tie_weights else self.nhid)).zero_(),
                     weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (
                         self.ninp if self.tie_weights else self.nhid)).zero_())
                    for l in range(self.nlayers)]
        elif self.rnn_type == 'QRNN' or self.rnn_type == 'GRU':
            return [weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (
                self.ninp if self.tie_weights else self.nhid)).zero_()
                    for l in range(self.nlayers)]

#
# import torch
# import torch.nn as nn
# from torch.autograd import Variable
#
# from embed_regularize import embedded_dropout
# from locked_dropout import LockedDropout
# from weight_drop import WeightDrop
#
# class RNNModel(nn.Module):
#     """Container module with an encoder, a recurrent module, and a decoder."""
#
#     def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, dropouth=0.5, dropouti=0.5, dropoute=0.1, wdrop=0, tie_weights=False):
#         super(RNNModel, self).__init__()
#         self.lockdrop = LockedDropout()
#         self.idrop = nn.Dropout(dropouti)
#         self.hdrop = nn.Dropout(dropouth)
#         self.drop = nn.Dropout(dropout)
#         self.encoder = nn.Embedding(ntoken, ninp)
#         assert rnn_type in ['LSTM', 'QRNN', 'GRU'], 'RNN type is not supported'
#         if rnn_type == 'LSTM':
#             self.rnns = [torch.nn.LSTM(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else (ninp if tie_weights else nhid), 1, dropout=0) for l in range(nlayers)]
#             if wdrop:
#                 self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop) for rnn in self.rnns]
#         if rnn_type == 'GRU':
#             self.rnns = [torch.nn.GRU(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else ninp, 1, dropout=0) for l in range(nlayers)]
#             if wdrop:
#                 self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop) for rnn in self.rnns]
#         elif rnn_type == 'QRNN':
#             from torchqrnn import QRNNLayer
#             self.rnns = [QRNNLayer(input_size=ninp if l == 0 else nhid, hidden_size=nhid if l != nlayers - 1 else (ninp if tie_weights else nhid), save_prev_x=True, zoneout=0, window=2 if l == 0 else 1, output_gate=True) for l in range(nlayers)]
#             for rnn in self.rnns:
#                 rnn.linear = WeightDrop(rnn.linear, ['weight'], dropout=wdrop)
#         print(self.rnns)
#         self.rnns = torch.nn.ModuleList(self.rnns)
#         self.decoder = nn.Linear(nhid, ntoken)
#
#         # Optionally tie weights as in:
#         # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
#         # https://arxiv.org/abs/1608.05859
#         # and
#         # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
#         # https://arxiv.org/abs/1611.01462
#         if tie_weights:
#             #if nhid != ninp:
#             #    raise ValueError('When using the tied flag, nhid must be equal to emsize')
#             self.decoder.weight = self.encoder.weight
#
#         self.init_weights()
#
#         self.rnn_type = rnn_type
#         self.ninp = ninp
#         self.nhid = nhid
#         self.nlayers = nlayers
#         self.dropout = dropout
#         self.dropouti = dropouti
#         self.dropouth = dropouth
#         self.dropoute = dropoute
#         self.tie_weights = tie_weights
#
#     def reset(self):
#         if self.rnn_type == 'QRNN': [r.reset() for r in self.rnns]
#
#     def init_weights(self):
#         initrange = 0.1
#         self.encoder.weight.data.uniform_(-initrange, initrange)
#         self.decoder.bias.data.fill_(0)
#         self.decoder.weight.data.uniform_(-initrange, initrange)
#
#     def forward(self, input, hidden, return_h=False):
#         emb = embedded_dropout(self.encoder, input, dropout=self.dropoute if self.training else 0)
#         #emb = self.idrop(emb)
#
#         emb = self.lockdrop(emb, self.dropouti)
#
#         raw_output = emb
#         new_hidden = []
#         #raw_output, hidden = self.rnn(emb, hidden)
#         raw_outputs = []
#         outputs = []
#         for l, rnn in enumerate(self.rnns):
#             current_input = raw_output
#             raw_output, new_h = rnn(raw_output, hidden[l])
#             new_hidden.append(new_h)
#             raw_outputs.append(raw_output)
#             if l != self.nlayers - 1:
#                 #self.hdrop(raw_output)
#                 raw_output = self.lockdrop(raw_output, self.dropouth)
#                 outputs.append(raw_output)
#         hidden = new_hidden
#
#         output = self.lockdrop(raw_output, self.dropout)
#         outputs.append(output)
#
#         decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
#         result = decoded.view(output.size(0), output.size(1), decoded.size(1))
#         if return_h:
#             return result, hidden, raw_outputs, outputs
#         return result, hidden
#
#     def init_hidden(self, bsz):
#         weight = next(self.parameters()).data
#         # print ('in var init hidden: ', bsz)
#         if self.rnn_type == 'LSTM':
#             return [(Variable(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_()),
#                     Variable(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_()))
#                     for l in range(self.nlayers)]
#         elif self.rnn_type == 'QRNN' or self.rnn_type == 'GRU':
#             return [Variable(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_())
#                     for l in range(self.nlayers)]
