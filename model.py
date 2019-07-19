import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import tqdm

if torch.cuda.is_available():
    def to_torch(x, dtype, req = False):
        tor_type = torch.cuda.LongTensor if dtype == "int" else torch.cuda.FloatTensor
        x = Variable(torch.from_numpy(x).type(tor_type), requires_grad = req)
        return x
else:
    def to_torch(x, dtype, req = False):
        tor_type = torch.LongTensor if dtype == "int" else torch.FloatTensor
        x = Variable(torch.from_numpy(x).type(tor_type), requires_grad = req)
        return x

class Draft(nn.Module):

    # init with (channel, height, width) and out_dim for classiication
    def __init__(self, feature_dim, emb_dim):
        super(Draft, self).__init__()
        self.name = "EMB"

        # for now emb_dim is n_hidden is same. might change later
        n_hidden = 128
        self.n_hidden = n_hidden

        self.emb = nn.Linear(feature_dim, emb_dim)
        self.emb_to_hidden = nn.Linear(emb_dim, n_hidden)

        # for the transformer network
        self.Q = nn.Linear(n_hidden, n_hidden)
        self.K = nn.Linear(n_hidden, n_hidden)
        self.V = nn.Linear(n_hidden, n_hidden)
        self.QK = nn.Linear(n_hidden + n_hidden, 1)
        self.A2N = nn.Linear(n_hidden, n_hidden)

        self.fc1 = nn.Linear(n_hidden, n_hidden)
        self.fc2 = nn.Linear(n_hidden, feature_dim)

        self.opt = torch.optim.RMSprop(self.parameters(), lr=0.0001)


    def communicate(self, nodes):
        nn = len(nodes)
        qq = [self.Q(node) for node in nodes]
        kk = [self.K(node) for node in nodes]
        vv = torch.stack([self.V(node) for node in nodes]).transpose(0,1)

        ret = []
        for q in qq:
            # put the q and k together and compute the weight softmax
            qks = [torch.cat((q,k),dim=1)  for k in kk]
            qk_weights = [(self.QK(qk)).squeeze(-1) for qk in qks]
            qk_w = F.softmax(torch.stack(qk_weights).transpose(0,1), dim=1)
            aggre = torch.sum(qk_w.unsqueeze(2).repeat(1,1,self.n_hidden) * vv, dim=1)
            node_new = F.relu(self.A2N(aggre)) 
            ret.append(node_new)
        return ret

    def forward(self, As):
        embA = [self.emb(A) for A in As]
        embA = [F.relu(self.emb_to_hidden(a)) for a in embA]
        nodes = embA
        for i in range(3):
            nodes = self.communicate(nodes)

        agg, _ = torch.max(torch.stack(nodes), dim=0)

        return F.log_softmax(self.fc2(F.relu(self.fc1(agg))), dim=1)

    def loss(self, Bs_pred, Bs):
        return -torch.sum(Bs_pred * Bs)
  
    def learn_once(self, As, Bs):
        As = [to_torch(A, "float") for A in As]
        Bs = to_torch(Bs, "float")

        # optimize 
        self.opt.zero_grad()
        Bs_pred = self(As)
        loss = self.loss(Bs_pred, Bs)
        loss.backward()
        self.opt.step()
  
        return loss

    def embed(self, X):
        X = to_torch(X, "float")
        return self.encode(X)

    def save(self, loc):
        torch.save(self.state_dict(), loc)

    def load(self, loc):
        self.load_state_dict(torch.load(loc))


if __name__ == '__main__':
    from process import gen_train_batch, L

    draft = Draft(L, 32).cuda()

    for i in tqdm.tqdm(range(10000000)):
        As,Bs = gen_train_batch(100)
        loss = draft.learn_once(As,Bs)
        if i % 100 == 0:
            print (loss, len(As), loss / (8 - len(As)))
            draft.save("draft.mdl")


