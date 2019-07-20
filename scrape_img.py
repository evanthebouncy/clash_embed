import requests
import json
import time
import pickle

teams = []

headers = {
    'authorization': 'Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiIsImtpZCI6IjI4YTMxOGY3LTAwMDAtYTFlYi03ZmExLTJjNzQzM2M2Y2NhNSJ9.eyJpc3MiOiJzdXBlcmNlbGwiLCJhdWQiOiJzdXBlcmNlbGw6Z2FtZWFwaSIsImp0aSI6IjU0N2U3NWVmLThmM2ItNDJmNy04NjJkLWJkNzAxNTU1NzQwYyIsImlhdCI6MTU2MzQ4Mjk0Miwic3ViIjoiZGV2ZWxvcGVyL2E4MDZkZjgzLWJhZDMtMjZhZC0zNWE3LTgzYzRjOWM5ZTgxNiIsInNjb3BlcyI6WyJyb3lhbGUiXSwibGltaXRzIjpbeyJ0aWVyIjoiZGV2ZWxvcGVyL3NpbHZlciIsInR5cGUiOiJ0aHJvdHRsaW5nIn0seyJjaWRycyI6WyIxMjguMzAuOTMuMjQxIl0sInR5cGUiOiJjbGllbnQifV19.kztFoiN2MPYYFNcad901w24gjRVAl0JXWAlZR85_VMm4Rhp2oOyZqkehcoQUS3dC5oWV9uBr06K0xp4KVTA2mQ',
}

# first grab all the location ids
def grab_cards():
    url = 'https://api.clashroyale.com/v1/cards'
    r = requests.get(url, headers = headers)
    cards = [(x['id'], x['iconUrls']['medium']) for x in r.json()['items']]
    return cards

cards = grab_cards()
print (cards)

import urllib.request
for card_id, card_url in cards:
    urllib.request.urlretrieve(card_url, f"assets/{card_id}.png")


