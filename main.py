import numpy as np
import torch as T
import os
from model import VAE
from dataset import get_iterators
from helper_functions import get_cuda, get_sentences_in_batch
import torch.nn.functional as F
import math
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--n_vocab', type=int, default=12000)
parser.add_argument('--epochs', type=int, default=121)
parser.add_argument('--n_hidden_G', type=int, default=512)
parser.add_argument('--n_layers_G', type=int, default=2)
parser.add_argument('--n_hidden_E', type=int, default=512)
parser.add_argument('--n_layers_E', type=int, default=1)
parser.add_argument('--n_z', type=int, default=100)
parser.add_argument('--word_dropout', type=float, default=0.5)
parser.add_argument('--rec_coef', type=float, default=7)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--gpu_device', type=int, default=1)
parser.add_argument('--n_highway_layers', type=int, default=2)
parser.add_argument('--n_embed', type=int, default=300)
parser.add_argument('--unk_token', type=str, default="<unk>")
parser.add_argument('--pad_token', type=str, default="<pad>")
parser.add_argument('--start_token', type=str, default="<sos>")
parser.add_argument('--end_token', type=str, default="<eos>")

def str2bool(v):
    if v.lower() == 'true':
        return True
    else:
        return False

parser.add_argument('--resume_training', type=str2bool, default=False)
parser.add_argument('--to_train', type=str2bool, default=True)


opt = parser.parse_args()
print(opt)
save_path = "data/saved_models/vae_model.tar"
if not os.path.exists("data/saved_models"):
    os.makedirs("data/saved_models")

os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu_device)    #str(gpu_device)

#---------------------------------------------------------------

train_iter, val_iter, vocab = get_iterators(opt)

vae = VAE(opt)
vae.embedding.weight.data.copy_(vocab.vectors)              #Intialize trainable embeddings with pretrained glove vectors

vae = get_cuda(vae)
trainer_vae = T.optim.Adam(vae.parameters(), lr=opt.lr)

def create_generator_input(x, train):
    G_inp = x[:, 0:x.size(1)-1].clone()	                    #input for generator should exclude last word of sequence
    if train == False:
        return G_inp

    r = np.random.rand(G_inp.size(0), G_inp.size(1))
                                                            #Perform word_dropout according to random values (r) generated for each word
    for i in range(len(G_inp)):
        for j in range(1,G_inp.size(1)):
            if r[i, j] < opt.word_dropout and G_inp[i, j] not in [vocab.stoi[opt.pad_token], vocab.stoi[opt.end_token]]:
                G_inp[i, j] = vocab.stoi[opt.unk_token]

    return G_inp

def train_batch(x, G_inp, step, train = True):
    logit, _, kld = vae(x, G_inp, None, None)
    logit = logit.view(-1, opt.n_vocab)	                    #converting into shape (batch_size*(n_seq-1), n_vocab) to facilitate performing F.cross_entropy()
    x = x[:, 1:x.size(1)]	                                #target for generator should exclude first word of sequence
    x = x.contiguous().view(-1)	                            #converting into shape (batch_size*(n_seq-1),1) to facilitate performing F.cross_entropy()
    rec_loss = F.cross_entropy(logit, x)
    kld_coef = (math.tanh((step - 15000)/1000) + 1) / 2
    # kld_coef = min(1,step/(200000.0))
    loss = opt.rec_coef*rec_loss + kld_coef*kld
    if train == True:	                                    #skip below step if we are performing validation
        trainer_vae.zero_grad()
        loss.backward()
        trainer_vae.step()
    return rec_loss.item(), kld.item()

def load_model_from_checkpoint():
    global vae, trainer_vae
    checkpoint = T.load(save_path)
    vae.load_state_dict(checkpoint['vae_dict'])
    trainer_vae.load_state_dict(checkpoint['vae_trainer'])
    return checkpoint['step'], checkpoint['epoch']

def training():
    start_epoch = step = 0
    if opt.resume_training:
        step, start_epoch = load_model_from_checkpoint()
    for epoch in range(start_epoch, opt.epochs):
        vae.train()
        train_rec_loss = []
        train_kl_loss = []
        for batch in train_iter:
            x = batch.text 	                                #Used as encoder input as well as target output for generator
            G_inp = create_generator_input(x, train = True)
            rec_loss, kl_loss = train_batch(get_cuda(x), get_cuda(G_inp), step, train=True)
            train_rec_loss.append(rec_loss)
            train_kl_loss.append(kl_loss)
            step += 1

        vae.eval()
        valid_rec_loss = []
        valid_kl_loss = []
        for batch in val_iter:
            x = batch.text
            G_inp = create_generator_input(x, train = False)
            with T.autograd.no_grad():
                rec_loss, kl_loss = train_batch(get_cuda(x), get_cuda(G_inp), step, train=False)
            valid_rec_loss.append(rec_loss)
            valid_kl_loss.append(kl_loss)

        train_rec_loss = np.mean(train_rec_loss)
        train_kl_loss = np.mean(train_kl_loss)
        valid_rec_loss = np.mean(valid_rec_loss)
        valid_kl_loss = np.mean(valid_kl_loss)

        print("No.", epoch, "T_rec:", '%.2f'%train_rec_loss, "T_kld:", '%.2f'%train_kl_loss, "V_rec:", '%.2f'%valid_rec_loss, "V_kld:", '%.2f'%valid_kl_loss)
        if epoch%5==0:
            T.save({
                'epoch': epoch + 1,
                'vae_dict': vae.state_dict(),
                'vae_trainer': trainer_vae.state_dict(),
                'step': step
            }, save_path)

def generate_sentences(n_examples):                             #Generate n sentences
    checkpoint = T.load(save_path)
    vae.load_state_dict(checkpoint['vae_dict'])
    vae.eval()
    del checkpoint
    for i in range(n_examples):
        z = get_cuda(T.randn([1,opt.n_z]))
        h_0 = get_cuda(T.zeros(opt.n_layers_G, 1, opt.n_hidden_G))
        c_0 = get_cuda(T.zeros(opt.n_layers_G, 1, opt.n_hidden_G))
        G_hidden = (h_0, c_0)
        G_inp = T.LongTensor(1,1).fill_(vocab.stoi[opt.start_token])
        G_inp = get_cuda(G_inp)
        str = opt.start_token+" "
        while G_inp[0][0].item() != vocab.stoi[opt.end_token]:
            with T.autograd.no_grad():
                logit, G_hidden, _ = vae(None, G_inp, z, G_hidden)
            probs = F.softmax(logit[0], dim=1)
            G_inp = T.multinomial(probs,1)
            str += (vocab.itos[G_inp[0][0].item()]+" ")
        print(str.encode('utf-8'))

if __name__ == '__main__':
    if opt.to_train:
        training()
    else:
        generate_sentences(50)
