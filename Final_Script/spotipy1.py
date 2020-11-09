import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import time 
import sys
import os

client_id = 'c91326a9294a4d66966c2623d5147715'
client_secret = 'c1c30fb6969947d7b933f1f6e243f1b0'

client_credentials_manager = SpotifyClientCredentials(client_id, client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

def getcsvdata():
  datafile = sys.argv[1]
  with open(datafile,"r",encoding='utf-8') as filename:
    arr=[]
    for line in filename.read().split('\n')[1:-1]:
      id = line.split(',')
      id = id[-1].split('/')[-1]
      arr.append(id)
  
  arr = list(set(arr))

  if os.path.exists("ids1.txt"):
    os.remove("ids1.txt") 
  with open("ids1.txt","a") as outputfile:
    for id in arr:
      if id=="":
        continue
      outputfile.write(id+"\n")

getcsvdata()

def getTrackIDs():
  
    with open("ids1.txt","r") as filename:
      id = filename.read().split("\n")
      
    return id[:-1]

ids = getTrackIDs()
print("generated ids")

def getTrackFeatures(id):
  try:
    meta = sp.track(id)

    features = sp.audio_features(id)

    # meta
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

    track = [name,album_name,artist_name,artist_id,album_release_date,album_type,available_markets,length,popularity,trackid,mode,key,acousticness,danceability,energy,instrumentalness,liveness,loudness,speechiness,valence,tempo,time_signature]

    
    return track   

  except:
    pass

print("getting features of each song")
  
# loop over track ids 
tracks = []
for i in range(len(ids)):
  # time.sleep(.5)
  track = getTrackFeatures(ids[i])
  tracks.append(track)

print(len(tracks))

df = pd.DataFrame(tracks, columns = ['name','album_name','artist_name','artist_id','album_release_date','album_type','available_markets','length','popularity','trackid','mode','key','acousticness','danceability','energy','instrumentalness','liveness','loudness','speechiness','valence','tempo','time_signature'])

outputfilename = sys.argv[2]
if os.path.exists(outputfilename):
  os.remove(outputfilename) 

df.to_csv(outputfilename, sep = ',')
print("completed for "+outputfilename)