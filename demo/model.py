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
        # self.convs1 = [nn.Conv2d(Ci, Co, (K, D)) for K in Ks]
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        '''
        self.conv13 = nn.Conv2d(Ci, Co, (3, D))
        self.conv14 = nn.Conv2d(Ci, Co, (4, D))
        self.conv15 = nn.Conv2d(Ci, Co, (5, D))
        '''
        self.fc1 = nn.Linear(len(Ks)*Co, C1)

        self.convs2 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, 50)) for K in Ks])
        '''
        self.conv13 = nn.Conv2d(Ci, Co, (3, D))
        self.conv14 = nn.Conv2d(Ci, Co, (4, D))
        self.conv15 = nn.Conv2d(Ci, Co, (5, D))
        '''
        self.dropout = nn.Dropout(args.dropout)
        self.fc2 = nn.Linear(len(Ks)*Co, C2)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        # print("1. ", x.shape)
        x = self.embed(x)  # (N, W, D)
        # print("2. ", x.shape)
        if self.args.static:
            x = Variable(x)
        # print("3. ", x.shape)
        x = x.unsqueeze(1)  # (N, Ci, W, D)
        # print("4. ", x.shape)
        # x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)
        # print("5. ", x.shape)
        x = [F.relu(conv(x)) for conv in self.convs1]
        # for i in x:
            # print("6. ", i.shape)
        x = [_x.squeeze(3) for _x in x]
        # for i in x:
            # print("7. ", i.shape)
        x1 = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        # for i in x1:
            # print("8. ", i.shape)
        x1 = torch.cat(x1, 1)
        # print("9. ", x1.shape)

        x2 = [_x.transpose(1,2).unsqueeze(1) for _x in x]
        # for i in x2:
            # print("13. ", i.shape)
        x2 = [F.max_pool2d(i, kernel_size=2) for i in x2]
        # for i in x2:
            # print("10. ", i.shape)
        # x2 = torch.cat(x2, 1)
        # print("11. ", x2.shape)
        x2 = [F.relu(conv(_x)).squeeze(3) for conv, _x in zip(self.convs2, x2)]
        # print(x2)
        # for i in x2:
            # print("12. ", i.shape)

        x2 = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        # for i in x2:
            # print("14. ", i.shape)
        x2 = torch.cat(x2, 1)
        # print("15. ", x2.shape)

        '''
        x1 = self.conv_and_pool(x,self.conv13) #(N,Co)
        x2 = self.conv_and_pool(x,self.conv14) #(N,Co)
        x3 = self.conv_and_pool(x,self.conv15) #(N,Co)
        x = torch.cat((x1, x2, x3), 1) # (N,len(Ks)*Co)
        '''
        x1 = self.dropout(x1)  # (N, len(Ks)*Co)
        logit1 = self.fc1(x1)  # (N, C)

        # print(logit1)

        x2 = self.dropout(x2)
        logit2 = self.fc2(x2)

        # print(logit2)

        return logit1, logit2
