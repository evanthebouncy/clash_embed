import pickle
import numpy as np
import random

# extract entities and encodings and such
decks = pickle.load(open("decks.p", "rb"))
id_decks = [[x[1] for x in deck] for deck in decks]

card_ids = set()
for deck in decks:
    for name, c_id in deck:
        card_ids.add((c_id, name))

card_ids = list(sorted(list(card_ids)))
id_2_name = dict(card_ids)
card_ids = [x[0] for x in card_ids]
L = len(card_ids)

card_2_1hot = dict([(card_ids[i],i) for i in range(L)])

def cards_to_vec(cards):
    ret = np.zeros((L,))
    for c in cards:
        ret[card_2_1hot[c]] = 1.0
    return ret

def gen_random_training(n):
    r_deck = random.choice(id_decks)
    if n > len(r_deck) - 1:
        return gen_random_training(n)
    A = np.random.choice(r_deck,n,replace=False)
    B = [x for x in r_deck if x not in A]
    if len(A) == 0 or len(B) == 0:
        return gen_random_training()
    return A, B

def gen_train_batch(n=10):
    random_card_num = random.choice([1,2,3,4,5,6,7])
    tr = [gen_random_training(random_card_num) for _ in range(n)]
    As = [[] for _ in range(random_card_num)]
    for A,B in tr:
        for i, aa in enumerate(A):
            As[i].append(cards_to_vec([aa]))
    As = [np.array(A) for A in As]
    Bs = np.array([cards_to_vec(x[1]) for x in tr])
    return As, Bs

if __name__ == '__main__':
    print (cards_to_vec(id_decks[0]))
    print (gen_random_training(3))
    bat_A, bat_B = gen_train_batch(200)
    import pdb; pdb.set_trace()
