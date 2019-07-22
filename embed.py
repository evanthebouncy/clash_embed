import numpy as np
from model_emb import EMB
from process import gen_train_batch, L, card_ids, cards_to_vec, id_2_name

emb_one = EMB(L, 32).cuda()
emb_one.load("saved_models/emb_single1.mdl")

X = np.array([cards_to_vec([x]) for x in card_ids])
e = emb_one.embed(X).detach().cpu().squeeze().numpy()
print (e)
print (e.shape)

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.manifold import TSNE
plt.rcParams['figure.dpi'] = 800


for i in range(10):
    e_tsne = TSNE(n_components=2).fit_transform(e)
    x = [x[0] for x in e_tsne]
    y = [x[1] for x in e_tsne]
    names = [id_2_name[x] for x in card_ids]

    minx, maxx = min(x), max(x)
    miny, maxy = min(y), max(y)

    l_x, l_y = maxx-minx, maxy-miny

# for xx,yy,name in zip(x,y,names):
#     xx = xx - minx
#     yy = yy - miny
#     plt.text(xx/l_x, yy/l_y, name, fontsize=4)

    for xx,yy,id_name in zip(x,y,card_ids):
        xx = xx - minx
        yy = yy - miny
        img = mpimg.imread(f'assets/{id_name}.png')
        plt.figimage(img, xx/l_x*5120*0.9, yy/l_y*3840*0.9)
        # plt.text(xx/l_x, yy/l_y, name, fontsize=4)


    plt.savefig(f'emb_images/emb_{i}.png')
    plt.clf()
