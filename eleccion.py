import numpy as np
from model import Draft
from process import gen_train_batch, L, card_ids, cards_to_vec, id_2_name

draft = Draft(L, 32).cuda()
draft.load("saved_models/draft2.mdl")

N_CARDS = 16

subset_cards = np.random.choice(card_ids,N_CARDS,replace=False)
X = np.array([cards_to_vec([x]) for x in subset_cards])
e = draft.embed(X).detach().cpu().squeeze().numpy()

def dist(id1, id2):
    return np.sum((e[id1] - e[id2]) ** 2)

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

# all_costs = [(get_cost(pairs), pairs) for pairs in all_pairs]
# found_set = sorted(all_costs)[0][1]

# ============== A GREEDY ALGORITHM =============
def find_smallest(ban_set):
    best, smallest = None, 999999
    for id1 in range(N_CARDS):
        for id2 in range(N_CARDS):
            if (id1 not in ban_set) and (id2 not in ban_set) and (id1 != id2):
                dd = dist(id1, id2)
                if dd < smallest:
                    smallest = dd
                    best = [id1, id2]
    return best, smallest

found_set = []
for i in range(N_CARDS//2):
    best, smol = find_smallest(found_set)
    found_set = found_set + best

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
plt.rcParams['figure.dpi'] = 800
plt.figure(figsize=(7,0.4))

for i, idx in enumerate(found_set):
    gapp = 50 if i % 2 == 0 else 0
    id_name = subset_cards[idx]
    img = mpimg.imread(f'assets/{id_name}.png')
    plt.figimage(img, 350 * i + gapp, 0)

plt.savefig(f'emb_images/pair_learned.png')
plt.clf()

for i, c_id in enumerate(subset_cards):
    gapp = 50 if i % 2 == 0 else 0
    img = mpimg.imread(f'assets/{c_id}.png')
    plt.figimage(img, 350 * i + gapp, 0)

plt.savefig(f'emb_images/pair_random.png')
plt.clf()

