import torch as T

def get_cuda(tensor):
    if T.cuda.is_available():
        tensor = tensor.cuda()
    return tensor

def get_sentences_in_batch(x, vocab):
    for sent in x:
        str1 = ""
        for word in sent:
            str1 += vocab.itos[word] + " "
        print(str1)