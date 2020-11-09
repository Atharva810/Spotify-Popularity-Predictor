import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import time 
import sys
import os
import random

client_id = 'c91326a9294a4d66966c2623d5147715'
client_secret = 'c1c30fb6969947d7b933f1f6e243f1b0'

client_credentials_manager = SpotifyClientCredentials(client_id, client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
track_results= set()



def random_Search():
    characters = 'abcdefghijklmnopqrstuvwxyz'
    randomCharacter = characters[random.randint(0,len(characters)-1)]
    randomSearch =""

    no = random.randint(0,1)
    if no==0:
        return randomCharacter + '%'
    else:
        return '%' + randomCharacter + '%'

country = "ee"
for j in range(40):
    try:
        print(f"j: {j}")
        char = random_Search()
        for i in range(0, 2000, 50):
            
            tracks = sp.search(char, limit=50, offset=i,type="track",market=country)
            for t in tracks['tracks']['items']:
                track_results.add(t["id"])
    except:
        break
# print(track_results)
print(f"Track length: {len(track_results)}")

if os.path.exists("ids.txt"):
    os.remove("ids.txt")
with open("ids.txt", "a") as outputfile:
    for id in track_results:
        if id =="":
            continue
        outputfile.write(id+"\n")


def getTrackIDs():
    with open("ids.txt", "r") as filename:
        id = filename.read().split("\n")
    return id[:-2]

ids = getTrackIDs()
print("generated ids")


def getTrackFeatures():
    df = pd.DataFrame(columns=['name', 'album_name', 'artist_name', 'artist_id', 'album_release_date', 'album_type', 'available_markets', 'length', 'popularity','trackid', 'mode', 'key', 'acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'valence', 'tempo', 'time_signature'])
    for id in ids:
        # print(id)
        try:
            meta = sp.track(id)
            # print(len(meta))
            features = sp.audio_features(id)
            # print(len(features))
        # if len(meta)!=0 and len(features)!=0:
            name = meta['name']

            album_name = meta['album']['name']
            artist_name = meta['album']['artists'][0]['name']

            artist_id = meta['album']['artists'][0]['id']

            album_release_date = meta['album']['release_date']
            album_type = meta['album']['album_type']

            available_markets = meta['available_markets']

            length = meta['duration_ms']
            popularity = meta['popularity']
            trackid = meta['id']

            # features

            mode = features[0]['mode']
            key = features[0]['key']
            acousticness = features[0]['acousticness']
            danceability = features[0]['danceability']
            energy = features[0]['energy']
            instrumentalness = features[0]['instrumentalness']
            liveness = features[0]['liveness']
            loudness = features[0]['loudness']
            speechiness = features[0]['speechiness']
            valence = features[0]['valence']
            tempo = features[0]['tempo']
            time_signature = features[0]['time_signature']

            # track = [name, album_name, artist_name, artist_id, album_release_date, album_type, available_markets, length, popularity, trackid,mode, key, acousticness, danceability, energy, instrumentalness, liveness, loudness, speechiness, valence, tempo, time_signature]

            df = df.append({"name": name, "album_name": album_name, "artist_name": artist_name, "artist_id": artist_id, "album_release_date": album_release_date, "album_type": album_type, "available_markets": available_markets, "length": length, "popularity": popularity, "trackid": trackid, "mode": mode, "key": key, "acousticness": acousticness, "danceability": danceability, "energy": energy, "instrumentalness": instrumentalness, "liveness": liveness, "loudness": loudness, "speechiness": speechiness, "valence": valence, "tempo": tempo, "time_signature":time_signature}, ignore_index=True)
            # print(len(df))
        # print("here")
            
            # return track
        except:
            pass

    outputfilename = country+".csv"
    if os.path.exists(outputfilename):
        os.remove(outputfilename)

    df.to_csv(outputfilename, sep=',')
    print("completed for "+outputfilename)




df = getTrackFeatures()

# loop over track ids
# tracks = []
# for i in range(len(ids)):
#     # time.sleep(.5)
#     # track = 
#     getTrackFeatures(ids[i])
    # tracks.append(track)

# print(len(tracks))

# outputfilename = sys.argv[1]

