import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class CNN_Text(nn.Module):
    
    def __init__(self, args):
        super(CNN_Text, self).__init__()
        self.args = args
        
        V = args.embed_num
        D = args.embed_dim
        C1 = args.class1_num
        C2 = args.class2_num
        Ci = 1
        Co = args.kernel_num
        Ks = args.kernel_sizes

        self.embed = nn.Embedding(V, D)
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        self.fc1 = nn.Linear(len(Ks)*Co, C1)

        self.convs2 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, 50)) for K in Ks])
        self.dropout = nn.Dropout(args.dropout)
        self.fc2 = nn.Linear(len(Ks)*Co, C2)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        x = self.embed(x)  # (N, W, D)
        if self.args.static:
            x = Variable(x)
        x = x.unsqueeze(1)  # (N, Ci, W, D)
        x = [F.relu(conv(x)) for conv in self.convs1]
        x = [_x.squeeze(3) for _x in x]
        
        x1 = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x1 = torch.cat(x1, 1)

        x2 = [_x.transpose(1,2).unsqueeze(1) for _x in x]
        x2 = [F.max_pool2d(i, kernel_size=2) for i in x2]
        x2 = [F.relu(conv(_x)).squeeze(3) for conv, _x in zip(self.convs2, x2)]
        x2 = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x2 = torch.cat(x2, 1)

        x1 = self.dropout(x1)  # (N, len(Ks)*Co)
        logit1 = self.fc1(x1)  # (N, C)

        x2 = self.dropout(x2)
        logit2 = self.fc2(x2)

        return logit1, logit2
