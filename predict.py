import os
import argparse
import datetime
import torch
import torchtext.data as data
import torchtext.datasets as datasets
import model
import train
import mydatasets
import torch.autograd as autograd
import dill

class _args():
    def __init__(self):
        pass
        
args = _args()
args.lr = 0.001
args.epochs=256
args.batch_size=64
args.log_interval=1
args.test_interval=100
args.save_interval=500
args.save_dir='snapshot'
args.early_stop=1000
args.save_best=True
args.shuffle=False
# model
args.dropout=0.5
args.max_norm=3.0
args.embed_dim=128
args.kernel_num=100
args.kernel_sizes='3,4,5'
args.static=False
# device
args.device=-1
args.no_cuda=False
args.snapsho=None
args.predict=None
args.test=False
args.run=False

with open("tfield", 'rb') as fr:
    text_field = dill.load(fr)
with open("l1field", 'rb') as fr:
    label1_field = dill.load(fr)
with open("l2field", 'rb') as fr:
    label2_field = dill.load(fr)

args.embed_num = len(text_field.vocab)
args.class1_num = len(label1_field.vocab) - 1
args.class2_num = len(label2_field.vocab) - 1
args.cuda = (not args.no_cuda) and torch.cuda.is_available(); del args.no_cuda
args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
args.save_dir = os.path.join(args.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

print("\nParameters:")
for attr, value in sorted(args.__dict__.items()):
    print("\t{}={}".format(attr.upper(), value))


model = model.CNN_Text(args)
model.load_state_dict(torch.load("best_steps_21000.pt", map_location=torch.device('cpu')))
model.eval()

def predict(text):
    assert isinstance(text, str)
    text = text_field.preprocess(text)
    text = [[text_field.vocab.stoi[x] for x in text]]
    x = torch.tensor(text)
    x = autograd.Variable(x)
    logit1, logit2 = model(x)
    _, predicted1 = torch.max(logit1, 1)
    _, predicted2 = torch.max(logit2, 1)
    return label1_field.vocab.itos[predicted1.data[0]+1], label2_field.vocab.itos[predicted2.data[0]+1]