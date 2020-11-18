import pygal
import json
import matplotlib.pyplot as plt

worldmap_chart = pygal.maps.world.World()
worldmap_chart.title = 'Country wise features'
features_set={'loudness':[],'acousticness':[],'tempo':[],'energy':[],'danceability':[],'liveness':[],'speechiness':[],'valence':[]}


json_file='countrywise_features.json'

json_object={}
with open(json_file,'r',encoding='utf-8') as file:
    for line in file:
        json_object=line

my_dict=json.loads(json_object)
counts={'loudness':0,'acousticness':0,'tempo':0,'energy':0,'danceability':0,'liveness':0,'speechiness':0,'valence':0}

for key,value in my_dict.items():
    features_set[value[0][0]].append(key)
    counts[value[0][0]]+=1

print(features_set)

worldmap_chart.add('loudness', features_set['loudness'])
worldmap_chart.add('acousticness', features_set['acousticness'])
worldmap_chart.add('tempo', features_set['tempo'])
worldmap_chart.add('energy', features_set['energy'])
worldmap_chart.add('danceability', features_set['danceability'])
worldmap_chart.add('liveness', features_set['liveness'])
worldmap_chart.add('speechiness', features_set['speechiness'])
worldmap_chart.add('valence', features_set['valence'])

keys=counts.keys()
values=counts.values()

plt.bar(keys,values)

plt.show()


worldmap_chart.render_to_file('features2.svg')