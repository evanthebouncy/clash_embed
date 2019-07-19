import numpy as np
from model import Emb
from process import gen_train_batch, L, card_ids, cards_to_vec, id_2_name

emb = Emb(L, 32).cuda()
emb.load("emb1.mdl")

X = np.array([cards_to_vec([x]) for x in card_ids])
e = emb.embed(X).detach().cpu().squeeze().numpy()
print (e)
print (e.shape)

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
e_tsne = TSNE(n_components=2).fit_transform(e)
x = [x[0] for x in e_tsne]
y = [x[1] for x in e_tsne]
names = [id_2_name[x] for x in card_ids]

minx, maxx = min(x), max(x)
miny, maxy = min(y), max(y)

l_x, l_y = maxx-minx, maxy-miny

for xx,yy,name in zip(x,y,names):
    xx = xx - minx
    yy = yy - miny
    plt.text(xx/l_x, yy/l_y, name, fontsize=8)
plt.savefig('emb_.png')
