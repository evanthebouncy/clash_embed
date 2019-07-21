import requests
import json
import time
import pickle

teams = []

headers = {
    'authorization': 'Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiIsImtpZCI6IjI4YTMxOGY3LTAwMDAtYTFlYi03ZmExLTJjNzQzM2M2Y2NhNSJ9.eyJpc3MiOiJzdXBlcmNlbGwiLCJhdWQiOiJzdXBlcmNlbGw6Z2FtZWFwaSIsImp0aSI6IjU0N2U3NWVmLThmM2ItNDJmNy04NjJkLWJkNzAxNTU1NzQwYyIsImlhdCI6MTU2MzQ4Mjk0Miwic3ViIjoiZGV2ZWxvcGVyL2E4MDZkZjgzLWJhZDMtMjZhZC0zNWE3LTgzYzRjOWM5ZTgxNiIsInNjb3BlcyI6WyJyb3lhbGUiXSwibGltaXRzIjpbeyJ0aWVyIjoiZGV2ZWxvcGVyL3NpbHZlciIsInR5cGUiOiJ0aHJvdHRsaW5nIn0seyJjaWRycyI6WyIxMjguMzAuOTMuMjQxIl0sInR5cGUiOiJjbGllbnQifV19.kztFoiN2MPYYFNcad901w24gjRVAl0JXWAlZR85_VMm4Rhp2oOyZqkehcoQUS3dC5oWV9uBr06K0xp4KVTA2mQ',
}

# first grab all the location ids
def grab_locations():
    url = 'https://api.clashroyale.com/v1/locations'
    r = requests.get(url, headers = headers)
    locations = [x['id'] for x in r.json()['items']]
    return locations

# locations = grab_locations()
# print ('locations ', locations)

def grab_clans(location):
    url = f'https://api.clashroyale.com/v1/locations/{location}/rankings/clans'
    r = requests.get(url, headers = headers)
    tags = [x['tag'] for x in r.json()['items']]
    return tags

# clan_tags = grab_clans(locations[0])
# print (len(clan_tags))
# print ('clan tags ', clan_tags)

def grab_clan_players(clan_tag):
    url = f'https://api.clashroyale.com/v1/clans/{clan_tag}/members'
    r = requests.get(url, headers = headers)
    tags = [x['tag'] for x in r.json()['items']]
    return tags

# player_tags = grab_clan_players(clan_tags[-1].replace("#",'%23'))
# print ('player tags ', player_tags)

def grab_player_cards(player_id):
    url = 'https://api.clashroyale.com/v1/players/' + player_id
    r = requests.get(url, headers = headers)
    player = r.json()
    if player['challengeMaxWins'] >= 12:
        print ("pro")
        return [(x['name'], x['id']) for x in player["currentDeck"]]
    else:
        print ("noob")
        return None


#deck = grab_player_cards(player_tags[-1].replace("#", '%23'))
#print ('player deck ', deck)
#assert 0

all_decks = []
for location in grab_locations():
    time.sleep(0.1)
    # top 100 clans
    for clan in grab_clans(location)[:100]:
        try:
            time.sleep(0.1)
            for player in grab_clan_players(clan.replace("#","%23")):
                time.sleep(0.1)
                cards = grab_player_cards(player.replace("#","%23"))
                if cards is not None:
                    all_decks.append(cards)
            print (len(all_decks), 'dumping !')
            pickle.dump(all_decks, open( "decks.p", "wb" ) )
        except:
            print ("oh well something was wrong, continue . . . ")


