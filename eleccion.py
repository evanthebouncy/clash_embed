import numpy as np
from model_emb import EMB
from process import gen_train_batch, L, card_ids, cards_to_vec, id_2_name

emb_one = EMB(L, 32).cuda()
emb_one.load("saved_models/emb_single1.mdl")

N_CARDS = 16

subset_cards = np.random.choice(card_ids,N_CARDS,replace=False)
X = np.array([cards_to_vec([x]) for x in subset_cards])
e = emb_one.embed(X).detach().cpu().squeeze().numpy()

def dist(id1, id2):
    return np.sum((e[id1] - e[id2]) ** 2)

# =============== COMBINATORIALLY COMPLEX ====================
# def generate_all_pairs(n):
# 
#     def helper(unpaired):
#         if unpaired == []:
#             return [[]]
#         else:
#             to_ret = []
#             first = unpaired[0]
#             for second in unpaired[1:]:
#                 rest = [x for x in unpaired if x not in [first, second]]
#                 rest_all_pairs = helper(rest)
#                 for rest_p in rest_all_pairs:
#                     to_ret.append([first, second] + rest_p)
#             return to_ret
# 
#     return helper(list(range(n)))
# 
# all_pairs = generate_all_pairs(N_CARDS)
# 
# def get_cost(xx):
#     paira = xx[::2]
#     pairb = xx[1:][::2]
#     pairss = list(zip(paira, pairb))
#     return sum([dist(*x) for x in pairss])
# 
# all_costs = [(get_cost(pairs), pairs) for pairs in all_pairs]
# found_set = sorted(all_costs)[0][1]

# ============== A GREEDY ALGORITHM =============
def get_pair_cost(ban_set):
    costs = []
    for id1 in range(N_CARDS):
        for id2 in range(N_CARDS):
            if (id1 not in ban_set) and (id2 not in ban_set) and (id1 != id2):
                dd = dist(id1, id2)
                costs.append((dd, [id1, id2]))
    return costs

best_set = []
for i in range(N_CARDS//2):
    cost_item = sorted(get_pair_cost(best_set))[0]
    best_set = best_set + cost_item[1]

worst_set = []
for i in range(N_CARDS//2):
    cost_item = sorted(get_pair_cost(worst_set))[-1]
    worst_set = worst_set + cost_item[1]

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
plt.rcParams['figure.dpi'] = 800
plt.figure(figsize=(7,0.4))

def plot_set(card_set, name):
    for i, idx in enumerate(card_set):
        gapp = 50 if i % 2 == 0 else 0
        id_name = subset_cards[idx]
        img = mpimg.imread(f'assets/{id_name}.png')
        plt.figimage(img, 350 * i + gapp, 0)

    plt.savefig(f'emb_images/{name}.png')
    plt.clf()

plot_set(best_set, 'best_set')
plot_set(worst_set, 'worst_set')
