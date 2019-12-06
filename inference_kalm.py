import argparse
import time
import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import model
from model import RNNModel
import data
import model as model_file

from utils import batchify, get_batch, repackage_hidden

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='data/recipe_ori/',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (LSTM, QRNN, GRU)')
parser.add_argument('--emsize', type=int, default=400,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=1150,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=30,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=50,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=70,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.4,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dropouth', type=float, default=0.3,
                    help='dropout for rnn layers (0 = no dropout)')
parser.add_argument('--dropouti', type=float, default=0.65,
                    help='dropout for input embedding layers (0 = no dropout)')
parser.add_argument('--dropoute', type=float, default=0.1,
                    help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument('--wdrop', type=float, default=0.5,
                    help='amount of weight dropout to apply to the RNN hidden to hidden matrix')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--nonmono', type=int, default=5,
                    help='random seed')
parser.add_argument('--cuda', action='store_false', default=False,
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
randomhash = ''.join(str(time.time()).split('.'))
parser.add_argument('--save', type=str, default='1575428923389924.pt',
                    help='path to save the final model')
parser.add_argument('--alpha', type=float, default=2,
                    help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
parser.add_argument('--beta', type=float, default=1,
                    help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
parser.add_argument('--wdecay', type=float, default=1.2e-6,
                    help='weight decay applied to all weights')
parser.add_argument('--optimizer', type=str, default='sgd',
                    help='optimizer to use (sgd, adam)')
parser.add_argument('--when', nargs="+", type=int, default=[-1],
                    help='When (which epochs) to divide the learning rate by 10 - accepts multiple')
args = parser.parse_args()


class Inferance():

    def __init__(self):

        self.mcq_wrd = ['chicken', 'bread', 'apple', 'milk', 'salt',
                        'tomato']
        # record = {corpus.dictionary.word2idx['chicken'] : [], corpus.dictionary.word2idx['bread'] : [], corpus.dictionary.word2idx['apple'] : [], corpus.dictionary.word2idx['milk'] : [], corpus.dictionary.word2idx['salt'] : [], corpus.dictionary.word2idx['tomato'] : []}
        self.record = {192: [], 398: [], 1437: [], 41: [], 70: [], 740: []}
        self.mcq_result = {192: [], 398: [], 1437: [], 41: [], 70: [], 740: []}
        self.corpus = data.Corpus(args.data)

        self.eval_batch_size = 16
        self.test_batch_size = 1
        self.train_data = batchify(self.corpus.train, args.batch_size, args)
        self.val_data = batchify(self.corpus.valid, self.eval_batch_size, args)
        self.test_data = batchify(self.corpus.test, self.test_batch_size, args)
        self.criterion = None

    def wordEmbedding(self,vocab,dim):
        embeddings = np.zeros([len(vocab), dim])
        with open("D:/Git/Embeddings/glove.6B/glove.6B.100d.txt", 'r') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                if word in vocab:
                    word_idx = vocab[word]
                    embeddings[word_idx] = vector
        return embeddings



    def evaluate_both(self,data_source, batch_size=10):
        self.model.eval()
        ntokens = len(self.corpus.dictionary)
        hidden = self.model.init_hidden(batch_size)
        # m = nn.Softmax()
        mcq_ids = [self.corpus.dictionary.word2idx[w] for w in self.mcq_wrd]

        for batch, i in enumerate(range(0, data_source.size(0) - 1, args.bptt)):

            data, targets = get_batch(data_source, i, args, evaluation=True)
            print(data)
            dataToPrint= [self.corpus.dictionary.idx2word[i.item()] for i in data]
            targetsToPrint = [self.corpus.dictionary.idx2word[i.item()] for i in targets]

            if (batch_size == 1):
                hidden = self.model.init_hidden(batch_size)

            output, hidden = self.model(data, hidden)
            output_flat = output.view(-1, ntokens)
            output_flat_cb = output_flat.clone()

            #########
            temp_output = output_flat_cb.clone()

            val, keys_t = temp_output.data.max(1)
            prob_temp_output = nn.Softmax(temp_output)

            for i in range(len(targets.data)):
                w = targets[i].item()
                voilated = 0
                base = temp_output[i][w].item()
                if w in mcq_ids:
                    r = 0
                    r2 = 0
                    pred = keys_t[i]
                    if pred == w:
                        r = 1
                    for idd in mcq_ids:
                        if idd != w:
                            if base < temp_output[i][idd].item():
                                voilated = 1
                                break

                    self.record[w].append(r)
                    if voilated == 0: r2 = 1
                    self.mcq_result[w].append(r2)

            hidden = repackage_hidden(hidden)

        for idd in self.record:
            if len(self.record[idd]) > 0:
                print(self.corpus.dictionary.idx2word[idd], ' acc: ', sum(self.record[idd]), ' out of ', len(self.record[idd]),
                      sum(self.record[idd]) * 100.0 / len(self.record[idd]))
                print(self.corpus.dictionary.idx2word[idd], ' mcq acc: ', sum(self.mcq_result[idd]), ' out of ',
                      len(self.mcq_result[idd]),
                      sum(self.mcq_result[idd]) * 100.0 / len(self.mcq_result[idd]))

    def infer(self):

        ntokens = len(self.corpus.dictionary)
        self.model = RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.dropouth,
                               args.dropouti, args.dropoute, args.wdrop)

        ###
        criterion = nn.CrossEntropyLoss()

        params = list(self.model.parameters()) + list(criterion.parameters())
        total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in params if x.size())
        print('Args:', args)
        print('Model total parameters:', total_params)

        # Load the best saved model.
        with open(args.save, 'rb') as f:
            # model.load_state_dict(torch.load(f))
            model, criterion, optimizer = torch.load(f)

        test_batch_size = 1
        self.evaluate_both(self.test_data, test_batch_size)
        print('=' * 165)
        print('| End of testing | ')
        print('=' * 165)


if __name__ == '__main__':
    inf = Inferance()
    inf.infer()
