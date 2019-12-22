import argparse
import time
import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import model_bkp1
import data
import model_bkp1 as model_file

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

mcq_wrd = ['chicken', 'bread', 'apple', 'milk', 'salt',
           'tomato']  # ch=6134, bread=3553, apple = 16, milk=4359, salt=10576, tomato=3965
# mcq_ids = [192, 398, 1437, 41, 70, 740]

# record = {corpus.dictionary.word2idx['chicken'] : [], corpus.dictionary.word2idx['bread'] : [], corpus.dictionary.word2idx['apple'] : [], corpus.dictionary.word2idx['milk'] : [], corpus.dictionary.word2idx['salt'] : [], corpus.dictionary.word2idx['tomato'] : []}
record = {192: [], 398: [], 1437: [], 41: [], 70: [], 740: []}
mcq_result = {192: [], 398: [], 1437: [], 41: [], 70: [], 740: []}
# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.set_device(0)
        torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################

corpus = data.Corpus(args.data)

eval_batch_size = 16
test_batch_size = 1
train_data = batchify(corpus.train, args.batch_size, args)
val_data = batchify(corpus.valid, eval_batch_size, args)
test_data = batchify(corpus.test, test_batch_size, args)

###################################################
# Build the model
###############################################################################
from splitcross import SplitCrossEntropyLoss

criterion = None

ntokens = len(corpus.dictionary)
model = model_bkp1.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.dropouth,
                            args.dropouti, args.dropoute, args.wdrop)

###
criterion = nn.CrossEntropyLoss()

params = list(model.parameters()) + list(criterion.parameters())
total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in params if x.size())
print('Args:', args)
print('Model total parameters:', total_params)


###############################################################################
# Testing code
###############################################################################


def evaluate(data_source, batch_size=10):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    if args.model == 'QRNN': model.reset()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(batch_size)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, args, evaluation=True)
        output, hidden = model(data, hidden)
        output = output.view(-1, ntokens)
        total_loss += len(data) * criterion(model.decoder.weight, model.decoder.bias, output, targets).data
        hidden = repackage_hidden(hidden)
    return total_loss.item() / len(data_source)


def evaluate_both(data_source, batch_size=10):
    # Turn on evaluation mode which disables dropout.
    # loss = evaluate(data_source,batch_size)

    model.eval()
    if args.model == 'QRNN':
        model.reset()

    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(batch_size)

    m = nn.Softmax()
    mcq_ids = [corpus.dictionary.word2idx[w] for w in mcq_wrd]

    for batch, i in enumerate(range(0, data_source.size(0) - 1, args.bptt)):

        data, targets = get_batch(data_source, i, args, evaluation=True)

        if (batch_size == 1):
            hidden = model.init_hidden(batch_size)

        output, hidden = model(data, hidden)

        output_flat = output.view(-1, ntokens)

        candidates = set([corpus.dictionary.idx2word[i.item()] for i in targets])
        candidates_ids = set([i.item() for i in targets])

        numwords = output_flat.size()[0]
        # symbol_table = get_symbol_table(targets, targets2)

        output_flat_cb = output_flat.clone()
        sums = []
        for idxx in range(numwords):
            for pos in candidates_ids:  # for all candidates

                var_prob = output_flat_cb.data[idxx][pos]
                new_prob1 = 2 * var_prob  # just to scale values, emperical


        #########
        temp_output = output_flat_cb.clone()
        # print("our model") 
        # or 
        temp_output_entity_composite = output_flat.clone()
        # temp_output_type = output_flat2.clone()
        # print("awd-st baseline")
        #########

        val, keys_t = temp_output.data.max(1)

        val_entity_composite, keys_t_entity_composite = temp_output_entity_composite.data.max(1)
        # val_type, keys_t_type = temp_output_type.data.max(1)

        prob_temp_output = m(temp_output)
        prob_temp_output_baseline = m(temp_output_entity_composite)
        # prob_temp_output_type = m(temp_output_type)

        prb_val, prb_keys = prob_temp_output.data.max(1)
        prb_val_entity_composite, prb_keys_entity_composite = prob_temp_output_baseline.data.max(1)
        # prb_val_type, prb_keys_type = prob_temp_output_type.data.max(1)

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

                record[w].append(r)
                if voilated == 0: r2 = 1
                mcq_result[w].append(r2)

        hidden = repackage_hidden(hidden)

    for idd in record:
        if len(record[idd]) > 0:
            print(corpus.dictionary.idx2word[idd], ' acc: ', sum(record[idd]), ' out of ', len(record[idd]),
                  sum(record[idd]) * 100.0 / len(record[idd]))
            print(corpus.dictionary.idx2word[idd], ' mcq acc: ', sum(mcq_result[idd]), ' out of ', len(mcq_result[idd]),
                  sum(mcq_result[idd]) * 100.0 / len(mcq_result[idd]))


# Load the best saved model.
with open(args.save, 'rb') as f:
    # model.load_state_dict(torch.load(f))
    model, criterion, optimizer = torch.load(f)

test_batch_size = 1
evaluate_both(test_data, test_batch_size)
print('=' * 165)
print('| End of testing | ')
print('=' * 165)
