import pickle
import sys
import timeit

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class CompoundProteinInteractionPrediction(nn.Module):
    def __init__(self):
        super(CompoundProteinInteractionPrediction, self).__init__()
        self.embed_fingerprint = nn.Embedding(n_fingerprint, dim)
        self.embed_word = nn.Embedding(n_word, dim)
        self.W_gnn = nn.ModuleList([nn.Linear(dim, dim)
                                    for _ in range(layer_gnn)])
        self.W_cnn = nn.ModuleList([nn.Conv2d(
                     in_channels=1, out_channels=1, kernel_size=2*window+1,
                     stride=1, padding=window) for _ in range(layer_cnn)])
        self.W_attention = nn.Linear(dim, dim)
        self.W_out = nn.Linear(2*dim, 2)

    def gnn(self, xs, A, layer):
        for i in range(layer):
            hs = torch.relu(self.W_gnn[i](xs))
            xs = xs + torch.matmul(A, hs)
        return torch.unsqueeze(torch.sum(xs, 0), 0)

    def cnn(self, xs, i):
        xs = torch.unsqueeze(torch.unsqueeze(xs, 0), 0)
        hs = torch.relu(self.W_cnn[i](xs))
        return torch.squeeze(torch.squeeze(hs, 0), 0)

    def attention_cnn(self, x, xs, layer):
        for i in range(layer):
            hs = self.cnn(xs, i)
            x = torch.relu(self.W_attention(x))
            hs = torch.relu(self.W_attention(hs))
            weights = torch.tanh(F.linear(x, hs))
            xs = torch.t(weights) * hs
        return torch.unsqueeze(torch.sum(xs, 0), 0)

    def forward(self, inputs):

        fingerprints, adjacency, words = inputs

        """Compound vector with GNN."""
        x_fingerprints = self.embed_fingerprint(fingerprints)
        x_compound = self.gnn(x_fingerprints, adjacency, layer_gnn)

        """Protein vector with attention-CNN."""
        x_words = self.embed_word(words)
        x_protein = self.attention_cnn(x_compound, x_words, layer_cnn)

        y_cat = torch.cat((x_compound, x_protein), 1)
        z_interaction = self.W_out(y_cat)

        return z_interaction

    def __call__(self, data):

        inputs = data[:]
        z_interaction = self.forward(inputs)

        z = F.softmax(z_interaction, 1).to('cpu').data[0].numpy()
        return z

def load_tensor(file_name, dtype):
    return [dtype(d).to(device) for d in np.load(file_name + '.npy')]


def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)

if __name__ == "__main__":

    (DATASET, radius, ngram, dim, layer_gnn, window, layer_cnn, lr, lr_decay,
     decay_interval, iteration, setting) = sys.argv[1:]

    (dim, layer_gnn, window, layer_cnn, decay_interval,
     iteration) = map(int, [dim, layer_gnn, window, layer_cnn, decay_interval,
                            iteration])
    lr, lr_decay = map(float, [lr, lr_decay])

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')

    dir_input = ('../dataset/' + DATASET + '/test/'
                 'radius' + radius + '_ngram' + ngram + '/')
    compounds = load_tensor(dir_input + 'compounds', torch.LongTensor)
    adjacencies = load_tensor(dir_input + 'adjacencies', torch.FloatTensor)
    proteins = load_tensor(dir_input + 'proteins', torch.LongTensor)

    dataset = list(zip(compounds, adjacencies, proteins))

    fingerprint_dict = load_pickle(dir_input + 'fingerprint_dict.pickle')
    word_dict = load_pickle(dir_input + 'word_dict.pickle')
    n_fingerprint = len(fingerprint_dict) + 1
    n_word = len(word_dict) + 1

    torch.manual_seed(1234)
    model = CompoundProteinInteractionPrediction()
    checkpoint = torch.load('../output/full_model/' + setting)
    model.load_state_dict(checkpoint)
    model = model.to(device)

    print('Testing Model')
    start = timeit.default_timer()

    

    for test_data in dataset:
        print(model(test_data)[1], '\t', model(test_data)[0])
