import re
from bs4 import BeautifulSoup
from pandas import Series,DataFrame
import pandas as pd

handle=open("Zhang_Shengxi_schedule.html")
soup=BeautifulSoup(handle,"html.parser")
file=str(soup)

file_a=str(soup("a"))
Classes=re.findall("\w+/.+/\d+\s-\s(.+?)</a>",file_a)

file_td=str(soup("td"))
Times_found=re.findall("(\w+)\s+(\d+:\d+\s\w+\s-\s\d+:\d+\s\w+)\s+;?",file_td)
#def deleteDuplicatedElementFromList(list):
#        return sorted(set(list), key = list.index)
#Times=deleteDuplicatedElementFromList(Times_found)
Times=Times_found[:len(Classes)]


file_td=str(soup("td"))
Rooms_found=re.findall("\w+\s+\d+:\d+\s\w+\s-\s\d+:\d+\s\w+\s+;\s+(.+)\s+</td>?",file_td)
#Rooms=deleteDuplicatedElementFromList(Rooms_found)
Rooms=Rooms_found[:len(Classes)]
for i in range(0,len(Rooms)):
    if Rooms[i]== "Arcadia Univ, , Room ":
        Rooms[i]="Unknown"


data={"Classes":Classes,"Times":Times,"Rooms":Rooms}
df=DataFrame(data)
print(df)
