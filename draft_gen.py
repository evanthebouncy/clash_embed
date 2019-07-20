import numpy as np
from model import Draft
from process import gen_train_batch, L, card_ids, cards_to_vec, id_2_name

draft = Draft(L, 32).cuda()
draft.load("saved_models/draft2.mdl")

deck = [np.random.choice(card_ids)]

As = [cards_to_vec(deck)]

for i in range(7):
    pr = draft.forward_pr(As)

    # mask off the chosen cards and re-normalise the rest
    masks_off = [card_ids.index(x) for x in deck]
    for idd in masks_off:
        pr[idd] = 0
    pr = pr / np.sum(pr)

    next_id = np.random.choice(list(range(len(card_ids))), p=pr)
    # next_id = np.argmax(pr)

    next_card_id = card_ids[next_id]
    deck = deck + [next_card_id]
    As = [cards_to_vec(deck)]

print ([id_2_name[x] for x in deck])
