import pygal
import json
import matplotlib.pyplot as plt
from pygal.style import Style
custom_style = Style(colors=('#00FF00', '#32CD32'))
# custom_style = Style(colors=('#00FFFF'))

worldmap_chart = pygal.maps.world.World(style=custom_style)
# worldmap_chart = pygal.maps.world.World()
worldmap_chart.title = 'Countrywise Accuracy'

accuracy_dict = {"95-100": [], "90-95": [], "85-90": [], "80-85": [], "75-80": [], "70-75": [],
                "65-70": [], "60-65": []}


json_file = 'countrywise_accuracy.json'

json_object = {}
with open(json_file, 'r', encoding='utf-8') as file:
    for line in file:
        json_object = line

my_dict = json.loads(json_object)
for k,v in my_dict.items():
    v=v*100
    if v >95 and v<=100:
        accuracy_dict["95-100"].append(k)
    if v >90 and v <=95:
        accuracy_dict["90-95"].append(k)
    if v > 85 and v <= 90:
        accuracy_dict["85-90"].append(k)
    if v > 80 and v <= 85:
        accuracy_dict["80-85"].append(k)
    if v > 75 and v <= 80:
        accuracy_dict["75-80"].append(k)
    if v > 70 and v <= 75:
        accuracy_dict["70-75"].append(k)
    if v > 65 and v <= 70:
        accuracy_dict["65-70"].append(k)
    if v > 60 and v <= 65:
        accuracy_dict["60-65"].append(k)

print(accuracy_dict)
# counts={'loudness':0,'acousticness':0,'tempo':0,'energy':0,'danceability':0,'liveness':0,'speechiness':0,'valence':0}

# counts = {"acousticness": 0, "danceability": 0, "energy": 0, "instrumentalness": 0, "key": 0, "liveness": 0,
#           "mode": 0, "loudness": 0, "speechiness": 0, "tempo": 0, "valence": 0, "length": 0, "time_signature": 0}

# for key, value in my_dict.items():
#     features_set[value[0][0]].append(key)
#     counts[value[0][0]] += 1

# print(

# worldmap_chart.add('60% to 65%', accuracy_dict["60-65"])
worldmap_chart.add('65% to 70%', accuracy_dict["65-70"])
worldmap_chart.add('70% to 75%', accuracy_dict["70-75"])
worldmap_chart.add('75% to 80%', accuracy_dict["75-80"])
worldmap_chart.add('80% to 85%', accuracy_dict["80-85"])
worldmap_chart.add('85% to 90%', accuracy_dict["85-90"])
worldmap_chart.add('90% to 95%', accuracy_dict["90-95"])
worldmap_chart.add("95% to 100%", accuracy_dict["95-100"])








# keys = counts.keys()
# values = counts.values()

# plt.bar(keys, values)

# plt.show()


worldmap_chart.render_to_file('accuracy_worldplot.svg')
