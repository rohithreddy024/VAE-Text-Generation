# -*- coding: utf-8 -*-

import torch as T
import torch.nn as nn
import torch.nn.functional as F
from helper_functions import get_cuda

class Highway(nn.Module):
    def __init__(self, opt):
        super(Highway, self).__init__()
        self.n_layers = opt.n_highway_layers
        self.non_linear = nn.ModuleList([nn.Linear(opt.n_embed, opt.n_embed) for _ in range(self.n_layers)])
        self.linear = nn.ModuleList([nn.Linear(opt.n_embed, opt.n_embed) for _ in range(self.n_layers)])
        self.gate = nn.ModuleList([nn.Linear(opt.n_embed, opt.n_embed) for _ in range(self.n_layers)])

    def forward(self, x):
        for layer in range(self.n_layers):
            gate = F.sigmoid(self.gate[layer](x))	        #Compute percentage of non linear information to be allowed for each element in x
            non_linear = F.relu(self.non_linear[layer](x))	#Compute non linear information
            linear = self.linear[layer](x)	                #Compute linear information
            x = gate*non_linear + (1-gate)*linear           #Combine non linear and linear information according to gate

        return x

class Encoder(nn.Module):
    def __init__(self, opt):
        super(Encoder, self).__init__()
        self.highway = Highway(opt)
        self.n_hidden_E = opt.n_hidden_E
        self.n_layers_E = opt.n_layers_E
        self.lstm = nn.LSTM(input_size=opt.n_embed, hidden_size=opt.n_hidden_E, num_layers=opt.n_layers_E, batch_first=True, bidirectional=True)

    def init_hidden(self, batch_size):
        h_0 = T.zeros(2*self.n_layers_E, batch_size, self.n_hidden_E)
        c_0 = T.zeros(2*self.n_layers_E, batch_size, self.n_hidden_E)
        self.hidden = (get_cuda(h_0), get_cuda(c_0))

    def forward(self, x):
        batch_size, n_seq, n_embed = x.size()
        x = self.highway(x)
        self.init_hidden(batch_size)
        _, (self.hidden, _) = self.lstm(x, self.hidden)	             #Exclude c_T and extract only h_T
        self.hidden = self.hidden.view(self.n_layers_E, 2, batch_size, self.n_hidden_E)
        self.hidden = self.hidden[-1]	                             #Select only the final layer of h_T
        e_hidden = T.cat(list(self.hidden), dim=1)	                 #merge hidden states of both directions; check size
        return e_hidden

class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        self.n_hidden_G = opt.n_hidden_G
        self.n_layers_G = opt.n_layers_G
        self.n_z = opt.n_z
        self.lstm = nn.LSTM(input_size=opt.n_embed+opt.n_z, hidden_size=opt.n_hidden_G, num_layers=opt.n_layers_G, batch_first=True)
        self.fc = nn.Linear(opt.n_hidden_G, opt.n_vocab)

    def init_hidden(self, batch_size):
        h_0 = T.zeros(self.n_layers_G, batch_size, self.n_hidden_G)
        c_0 = T.zeros(self.n_layers_G, batch_size, self.n_hidden_G)
        self.hidden = (get_cuda(h_0), get_cuda(c_0))

    def forward(self, x, z, g_hidden = None):
        batch_size, n_seq, n_embed = x.size()
        z = T.cat([z]*n_seq, 1).view(batch_size, n_seq, self.n_z)	#Replicate z inorder to append same z at each time step
        x = T.cat([x,z], dim=2)	                                    #Append z to generator word input at each time step

        if g_hidden is None:	                                    #if we are validating
            self.init_hidden(batch_size)
        else:					                                    #if we are training
            self.hidden = g_hidden

        #Get top layer of h_T at each time step and produce logit vector of vocabulary words
        output, self.hidden = self.lstm(x, self.hidden)
        output = self.fc(output)

        return output, self.hidden	                                #Also return complete (h_T, c_T) incase if we are testing


class VAE(nn.Module):
    def __init__(self, opt):
        super(VAE, self).__init__()
        self.embedding = nn.Embedding(opt.n_vocab, opt.n_embed)
        self.encoder = Encoder(opt)
        self.hidden_to_mu = nn.Linear(2*opt.n_hidden_E, opt.n_z)
        self.hidden_to_logvar = nn.Linear(2*opt.n_hidden_G, opt.n_z)
        self.generator = Generator(opt)
        self.n_z = opt.n_z

    def forward(self, x, G_inp, z = None, G_hidden = None):
        if z is None:	                                                #If we are testing with z sampled from random noise
            batch_size, n_seq = x.size()
            x = self.embedding(x)	                                    #Produce embeddings from encoder input
            E_hidden = self.encoder(x)	                                #Get h_T of Encoder
            mu = self.hidden_to_mu(E_hidden)	                        #Get mean of lantent z
            logvar = self.hidden_to_logvar(E_hidden)	                #Get log variance of latent z
            z = get_cuda(T.randn([batch_size, self.n_z]))	                #Noise sampled from ε ~ Normal(0,1)
            z = mu + z*T.exp(0.5*logvar)	                            #Reparameterization trick: Sample z = μ + ε*σ for backpropogation
            kld = -0.5*T.sum(logvar-mu.pow(2)-logvar.exp()+1, 1).mean()	#Compute KL divergence loss
        else:
            kld = None                                                  #If we are training with given text

        G_inp = self.embedding(G_inp)	                                #Produce embeddings for generator input

        logit, G_hidden = self.generator(G_inp, z, G_hidden)
        return logit, G_hidden, kld
