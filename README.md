# Clash Embeddings

For what this is on a high level, see the medium post :

https://medium.com/@evanthebouncy/card-similarities-in-clash-royal-80188d22d5b7

# Files and descriptions 

assets/ : stores the image files for card icons

decks\_200k.p : a pickle file of 200k competitive clash decks scrapped

scrape, scrape\_img : boilerplate code for scraping the clash API for decks and deck images

process : generate the right data for learning

model : a draft gnerative model

model\_emb : the 7 to 1 guessing game model

saved\_models/ : saved models, after a night of training on google cloud

embed : take the learned embedding and embed them

eleccion : take the learned embedding and pair cards together in a sensible way

draft\_gen : take the draft generative model and sample a random deck that "made sense"
