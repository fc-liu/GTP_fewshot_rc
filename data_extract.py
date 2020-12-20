import json
import os
from bs4 import BeautifulSoup

soup=BeautifulSoup(open(os.path.join("data/wiki_property","wiki_prop.html")))

prop_dict={}
prop_list=soup.find("table",{"class":"wikitable sortable"}).findAll("tr")

for prop in prop_list:
    tds=prop.findAll("td")
    if not tds:
        continue
    prop_dict[tds[0].text]={"label":tds[1].text,"description":tds[2].text.strip(),"alias":tds[3].text.strip(),"data_type":tds[4].text.strip(),"count":tds[5].text.strip()}
fewrel_prop={}
for file_name in ["train","val","test"]:
    fewrel_prop[file_name]={}
    file_path= os.path.join("data",file_name+".json")
    json_data = json.load(open(file_path, "r"))
    # self.data = {}
    classes = list(json_data.keys())
    for rel_id in classes:
        fewrel_prop[file_name][rel_id]=prop_dict[rel_id]['label']
    print(file_name)
    print(fewrel_prop[file_name])

print(fewrel_prop)