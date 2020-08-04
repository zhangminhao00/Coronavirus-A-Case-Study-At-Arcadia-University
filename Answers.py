import json
import networkx as nx
from networkx.algorithms import bipartite
import matplotlib.pyplot as plt
import operator
import re
import numpy as np
import pandas as pd
from collections import defaultdict

file=open("data_dump_1973.json")
file=json.load(file)       # convert into dic

f=open("data.txt","w")

G=nx.Graph()
student_index=list()
classes_name=list()
student_classes=list()
for i in file:
    if file[i]["Classes"]!=[]:
        student_index.append(i)
        for classes in file[i]["Classes"]:
            student_classes.append((i,classes))
            if classes in classes_name:
                continue
            classes_name.append(classes)
# print(len(classes_name)==len(set(classes_name)))
G.add_nodes_from(student_index,bipartite=0)
G.add_nodes_from(classes_name,bipartite=1)
# print(bipartite.is_bipartite(G))
#print(G.nodes())
G.add_edges_from(student_classes)
#print(G.edges())

P=bipartite.weighted_projected_graph(G,student_index)
# print(P.edges(data=True))
N_sameclasses=dict()
for i in list(P.edges(data=True)):
    weight=i[2]["weight"]
    if weight in N_sameclasses:
        N_sameclasses[weight]=N_sameclasses[weight]+1
    else:
        N_sameclasses[weight]=1
print("N_sameclasses:\n",N_sameclasses,file=f)
sum=sum(N_sameclasses.values())
N_sameclasses_prob=dict((k,v/sum) for k,v in N_sameclasses.items())
print("\n\n\nN_sameclasses_prob:\n",N_sameclasses_prob,file=f)

P_components_number = nx.number_connected_components(P)
P_components=sorted(nx.connected_components(P))
print("\n\n\nP_components_number:\n",P_components_number,file=f)
print("\n\n\nP_components:\n",P_components,file=f)

N_steps_students=list()
N_steps_students_average=list()
for i in P_components:
    subgraph_distance=list()
    P_subgraph=P.subgraph(i).copy()
    all_pairs = dict(nx.all_pairs_shortest_path_length(P_subgraph))
    for node1 in all_pairs:
        for node2 in all_pairs[node1]:
            subgraph_distance.append(all_pairs[node1][node2])
    N_steps_students.append(subgraph_distance)
    N_steps_students_average.append(nx.average_shortest_path_length(P_subgraph))
N_steps_students_statistics=list()
for i in N_steps_students:
    subgraph=dict()
    for j in i:
        if j in subgraph:
            subgraph[j]=subgraph[j]+1
        else:
            subgraph[j]=1
    #subgraph[j]=subgraph.get(j,0)+1
    N_steps_students_statistics.append(subgraph)
print("\n\n\nN_steps_students_statistics:\n",N_steps_students_statistics,file=f)
print("\n\n\nN_steps_students_average:\n",N_steps_students_average,file=f)

classes_node_size=[3*G.degree(v) for v in classes_name]
nx.draw_networkx_nodes(G,nodelist=student_index, pos = nx.random_layout(G),with_labels=False, node_color = "yellow",node_shape="o", edge_color = "red",width=0.05,node_size =3,label="Students")
nx.draw_networkx_nodes(G,nodelist=classes_name, pos = nx.random_layout(G),with_labels=False, node_color = "blue",node_shape="*", edge_color = "red",width=0.05,node_size = classes_node_size,label="Classes")
plt.legend(loc="lower right",facecolor='grey')
nx.draw_networkx_edges(G, pos = nx.random_layout(G),with_labels=False, edge_color = "red",width=0.1,label="Student-Class")
plt.title("Bipartite graph of students and classes",size=20)
ax1=plt.axes()
ax1.set_facecolor("thistle")
plt.show()
nx.draw_networkx_nodes(P,pos = nx.random_layout(P),with_labels=False, node_color = "yellow", edge_color = "red",width=0.01,node_size =5,label="Students")
plt.legend(loc="lower right",facecolor='grey')
nx.draw_networkx_edges(P, pos = nx.random_layout(P),with_labels=False, edge_color = "red",width=0.03,label="Student-Student")
plt.title("Projected bipartite graph of students",size=20)
ax2=plt.axes()
ax2.set_facecolor("lavender")
plt.show()

clustering = nx.average_clustering(P)
transitivity = nx.transitivity(P)
print("\n\n\nclustering:\n",clustering,file=f)
print("\n\n\ntransitivity:\n",transitivity,file=f)

degCent=nx.degree_centrality(G)
degCent_classes=dict((key,value) for key,value in degCent.items() if not re.match("\d+",key))
sorted_degCent=sorted(degCent_classes.items(),key=operator.itemgetter(1),reverse=True)
# # print(sorted_degCent)
#
closeCent=nx.closeness_centrality(G,wf_improved=True)
closeCent_classes=dict((key,value) for key,value in closeCent.items() if not re.match("\d+",key))
sorted_closeCent=sorted(closeCent_classes.items(),key=operator.itemgetter(1),reverse=True)
# # print(sorted_closeCent)
#
btwnCent=nx.betweenness_centrality(G,normalized=True,endpoints=False)
btwnCent_classes=dict((key,value) for key,value in btwnCent.items() if not re.match("\d+",key))
sorted_btwnCent=sorted(btwnCent_classes.items(),key=operator.itemgetter(1),reverse=True)
# # print(sorted_btwnCent)


##1
P_degree=[]
for i in P:
    P_degree.append(P.degree(i))
print("\n\n\nAnswer 1 :",file=f)
print("\nP_degree = ", P_degree, file=f)
print("\nmax : ",max(P_degree),file=f)
print("\nmean : ",np.mean(P_degree),file=f)

##2
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
Cent1=pd.DataFrame(sorted_degCent,columns=["Classes","degCent"])
Cent2=pd.DataFrame(sorted_closeCent,columns=["Classes","closeCent"])
Cent3=pd.DataFrame(sorted_btwnCent,columns=["Classes","btwnCent"])
Cent12=pd.merge(Cent1,Cent2,on="Classes")
Cent=pd.merge(Cent12,Cent3,on="Classes")
Cent["rectifiedCent"]=Cent["degCent"]/Cent["degCent"].mean()+Cent["closeCent"]/Cent["closeCent"].mean()
+Cent["btwnCent"]/Cent["btwnCent"].mean()
Cent=Cent.sort_values(by="rectifiedCent",ascending=False)
Cent=Cent.set_index("Classes")
print("\n\n\nAnswer 2:",file=f)
print("\n",Cent,file=f)

##3
classes_nodes={i for i,j in G.nodes(data=True) if j["bipartite"]==1}
classes_degree={}
for i in classes_nodes:
    classes_degree[i]=G.degree(i)
print("\n\n\nAnswer3:",file=f)
print("\nclasses degree : ", classes_degree,file=f)
print("\naverage of classes degree : ",np.mean(list(classes_degree.values())),file=f)

##4
Rooms_Dates={}
Rooms_Dates=defaultdict(list)
for i in file:
    for j in file[i]["Rooms"]:
        if j != "Unknown" and j != "Online Course, Online Course, Room Online" and j !="Online Course, Online Course, Room SYNC":
            if file[i]["Times"][file[i]["Rooms"].index(j)][0]!="Arranged":
                Rooms_Dates[j].append(file[i]["Times"][file[i]["Rooms"].index(j)][0])
# print(Rooms_Dates)
Monday=[]
Tuesday=[]
Wednesday=[]
Thursday=[]
Friday=[]
for i in Rooms_Dates:
    Monday.append(re.findall("M",str(Rooms_Dates[i])))
    Tuesday.append(re.findall("T",str(Rooms_Dates[i])))
    Wednesday.append(re.findall("W",str(Rooms_Dates[i])))
    Thursday.append(re.findall("R",str(Rooms_Dates[i])))
    Friday.append(re.findall("F",str(Rooms_Dates[i])))
Monday_n=[]
Tuesday_n=[]
Wednesday_n=[]
Thursday_n=[]
Friday_n=[]
Rooms=[]
for i in range(0,len(Monday)):
    Monday_n.append(len(Monday[i]))
for i in range(0,len(Tuesday)):
    Tuesday_n.append(len(Tuesday[i]))
for i in range(0,len(Wednesday)):
    Wednesday_n.append(len(Wednesday[i]))
for i in range(0,len(Thursday)):
    Thursday_n.append(len(Thursday[i]))
for i in range(0,len(Friday)):
    Friday_n.append(len(Friday[i]))
for i in Rooms_Dates.keys():
    Rooms.append(i)
data_Rooms_Dates={"Rooms":Rooms,"Monday":Monday_n,"Tuesday":Tuesday_n,"Wednesday":Wednesday_n,
"Thursday":Thursday_n,"Friday":Friday_n}
df_Rooms_Dates=pd.DataFrame(data_Rooms_Dates)
df_Rooms_Dates=df_Rooms_Dates.set_index("Rooms")
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
print("\n\n\nAnswer4:",file=f)
print("\nDataframe(Rooms-Dates):\n",df_Rooms_Dates,file=f)
print("\nMaximum flow rate on Monday is in the classroom ",
df_Rooms_Dates["Monday"].idxmax(),"and the flow rate is ",max(df_Rooms_Dates["Monday"]),file=f)
print("\nMaximum flow rate on Tuesday is in the classroom ",
df_Rooms_Dates["Tuesday"].idxmax(),"and the flow rate is ",max(df_Rooms_Dates["Tuesday"]),file=f)
print("\nMaximum flow rate on Wednesday is in the classroom ",
df_Rooms_Dates["Wednesday"].idxmax(),"and the flow rate is ",max(df_Rooms_Dates["Wednesday"]),file=f)
print("\nMaximum flow rate on Thursday is in the classroom ",
df_Rooms_Dates["Thursday"].idxmax(),"and the flow rate is ",max(df_Rooms_Dates["Thursday"]),file=f)
print("\nMaximum flow rate on Friday is in the classroom ",
df_Rooms_Dates["Friday"].idxmax(),"and the flow rate is ",max(df_Rooms_Dates["Friday"]),file=f)

##5
M={}
T={}
W={}
R={}
F={}
M=defaultdict(list)
T=defaultdict(list)
W=defaultdict(list)
R=defaultdict(list)
F=defaultdict(list)
for i in file:
    for j in file[i]["Times"]:
        if re.match(".*M.*",j[0]):
            M[i].append(j[1])
        if re.match(".*T.*",j[0]):
            T[i].append(j[1])
        if re.match(".*W.*",j[0]):
            W[i].append(j[1])
        if re.match(".*R.*",j[0]):
            R[i].append(j[1])
        if re.match(".*F.*",j[0]):
            F[i].append(j[1])
print("\n\n\nAnswer5:",file=f)
print("\nThe class time for each student on work days:","\nMonday:\n",M,"\nTuesday:\n",T,
"\nWednesday:\n",W,"\nThursday:\n",R,"\nFriday:\n",F,file=f)
Lunch_M={}
Lunch_T={}
Lunch_W={}
Lunch_R={}
Lunch_F={}
for i in M:
    if len(M[i])>1:
        if re.match(".*AM.*PM.*", str(M[i])) or re.match(".*PM.*AM.*", str(M[i])):
            Lunch_M[i]="Yes"
        else:
            Lunch_M[i]="No"
for i in T:
    if len(T[i])>1:
        if re.match(".*AM.*PM.*", str(T[i])) or re.match(".*PM.*AM.*", str(T[i])):
            Lunch_T[i]="Yes"
        else:
            Lunch_T[i]="No"
for i in W:
    if len(W[i])>1:
        if re.match(".*AM.*PM.*", str(W[i])) or re.match(".*PM.*AM.*", str(W[i])):
            Lunch_W[i]="Yes"
        else:
            Lunch_W[i]="No"
for i in R:
    if len(R[i])>1:
        if re.match(".*AM.*PM.*", str(R[i])) or re.match(".*PM.*AM.*", str(R[i])):
            Lunch_R[i]="Yes"
        else:
            Lunch_R[i]="No"
for i in F:
    if len(F[i])>1:
        if re.match(".*AM.*PM.*", str(F[i])) or re.match(".*PM.*AM.*", str(F[i])):
            Lunch_F[i]="Yes"
        else:
            Lunch_F[i]="No"
print("\n\n\nExtra time to have lunch at school on work days (Yes or No):","\nMonday:\n",Lunch_M,
"\nTuesday:\n",Lunch_T,"\nWednesday:\n",Lunch_W,"\nThursday:\n",Lunch_R,"\nFriday:\n",Lunch_F,file=f)

##6
connect_directly=len(P.edges())
print("\n\n\nAnswer6:",file=f)
print("\nthe pair of students connecting directly:\n",connect_directly,file=f)
total=0
for i in P_components:
    length=len(i)
    total=total+length*(length-1)/2
print("\nthe pair of students connecting in total:\n",total,file=f)

##7
G_test=G.copy()
G_test.remove_node("FT100/Lecture/1")
G_test.remove_node("FA102L/Lecture/1")
G_test.remove_node("BI242/Lecture/2")
G_test.remove_node("BI204/Lecture/2")
G_test.remove_node("CH201/Lecture/2")
P_test=bipartite.weighted_projected_graph(G_test,student_index)
print("\n\n\nAnswer7:",file=f)
print("\n Clustering before removing the high centrality class:\n",nx.average_clustering(P),file=f)
print("\n Clustering after removing the high centrality class:\n",nx.average_clustering(P_test),file=f)

##8
G_components=sorted(nx.connected_components(G))
node_cut=[]
for i in G_components:
    G_subgraph=G.subgraph(i).copy()
    node_cut.append(nx.minimum_node_cut(G_subgraph))
print("\n\n\nAnswer8:",file=f)
print("\n minimum nodes needing to be removed in order to cut the transmission:\n",node_cut,file=f)
