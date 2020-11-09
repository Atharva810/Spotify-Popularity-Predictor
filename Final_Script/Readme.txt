Prereqs-
pandas
pip install pandas
spotipy - 
pip install spotipy
if still doesnt work then run python setup.py file of the folder

Format of running spotify.py(Scraper Script which fetches top 200 songs as per region) is-
python spotify.py start_date end_date region
we will keep start_date as-2016-12-23 and end_date as 2020-09-11

Format of running spotipy.py(script which gets audio features and track information from spotify api) is-
python spotipy.py <input file that contains id> <output file to be created>
eg- python spotipy.py data\us.csv dataset\US.csv

Regions- 59 countries

Rupali- 

"us", "gb", "ad", "ar", "at", "au", "be", "bg", "bo", "br",
"ca", "ch", "cl", "co", "cr", "cy", "cz", "de", "dk", 

Atharva-

"ec", "ee", "es", "fi", "fr", "gr", "gt", "hk", "hn", "hu",
"id", "ie", "is", "it", "jp", "lt", "lu", "lv", "mc", "mt", 

Gayatri-

"mx", "my", "ni", "nl", "no", "nz", "pa", "pe", "ph", "pl",
"pt", "py", "se", "sg", "sk", "sv", "tr", "tw", "uy", "do"
