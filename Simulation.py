import json
import networkx as nx
from networkx.algorithms import bipartite
import matplotlib.pyplot as plt
import operator
import re
import numpy as np
import pandas as pd
from collections import defaultdict
import scipy.integrate

file=open("data_dump_1973.json")
file=json.load(file)

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
G.add_nodes_from(student_index,bipartite=0)
G.add_nodes_from(classes_name,bipartite=1)
# print(G.nodes())
G.add_edges_from(student_classes)
#print(G.edges())

P=bipartite.weighted_projected_graph(G,student_index)

P_components_number = nx.number_connected_components(P)
P_components=sorted(nx.connected_components(P))
# print(P_components_number)
# print(P_components)


#### SIMULATION #####

# SIR model
def SIR_model(y,t,N,beta,gamma):
    S,I,R= y
    dS_dt= -beta*S*I/N
    dI_dt= beta*S*I/N-gamma*I
    dR_dt= gamma*I
    return([dS_dt,dI_dt,dR_dt])

print("Enter the R0 value (0.0 ~ 5.7) for SIR_model:")
k=0
while k==0:
    try:
        R0_=float(input())
        if 0.0<=R0_ and R0_<=5.7:
            k=1
        else:
            print("R0 should be a value from 0.0 to 5.7")
            print("Enter the R0 value (0.0 ~ 5.7) for SIR_model:")
    except:
        print("R0 should be a value from 0.0 to 5.7")
        print("Enter the R0 value (0.0 ~ 5.7) for SIR_model:")
N=1861
I0=1861*688/100000
R0=0
S0=N-I0-R0
gamma=1/7   ##mean recovery rate (1/recovery period)
beta=R0_*gamma  ##infectious rate

t= np.linspace(0,80,80)
result1= scipy.integrate.odeint(SIR_model,[S0,I0,R0],t,args=(N,beta,gamma))
result1=np.array(result1)

plt.figure()
plt.subplot(121)
plt.plot(t,result1[:,0],label="Susceptible")
plt.plot(t,result1[:,1],label="Infectious")
plt.plot(t,result1[:,2],label="Recovered")
plt.vlines(15,-50,1800,colors="c",linestyles = "dashed",label="last day of fall quarter")
plt.legend(loc="best")
plt.grid()
plt.xlabel("week")
plt.ylabel("student numbers")
plt.title("SIR model simulation\n (The chosen R0 value is {})".format(R0_))


# SEIR model
def SEIR_model(y,t,N,beta,sigma,gamma):
    S,E,I,R= y
    dS_dt= -beta*S*I/N
    dE_dt= beta*S*I/N-sigma*E
    dI_dt= sigma*E-gamma*I
    dR_dt= gamma*I
    return([dS_dt,dE_dt,dI_dt,dR_dt])

print("Enter the R0 value (0.0 ~ 5.7) for SEIR_model:")
k=0
while k==0:
    try:
        R0_=float(input())
        if 0.0<=R0_ and R0_<=5.7:
            k=1
        else:
            print("R0 should be a value from 0.0 to 5.7")
            print("Enter the R0 value (0.0 ~ 5.7) for SEIR_model:")
    except:
        print("R0 should be a value from 0.0 to 5.7")
        print("Enter the R0 value (0.0 ~ 5.7) for SEIR_model:")

N=1861
E0=0
I0=1861*688/100000
R0=0
S0=N-I0-E0-R0
sigma=1/5.2*7 #incubation rate (1/latent period)
gamma=1/7   ##mean recovery rate (1/recovery period)
beta=R0_*gamma  ##infectious rate

t= np.linspace(0,80,80)
result2= scipy.integrate.odeint(SEIR_model,[S0,E0,I0,R0],t,args=(N,beta,sigma,gamma))
result2=np.array(result2)
plt.subplot(122)
plt.plot(t,result2[:,0],label="Susceptible")
plt.plot(t,result2[:,2],label="Infectious")
plt.plot(t,result2[:,3],label="Recovered")
plt.plot(t,result2[:,1],label="Exposed")
plt.vlines(15,-50,1800,colors="c",linestyles = "dashed",label="last day of fall quarter")
plt.legend(loc="best")
plt.grid()
plt.xlabel("week")
plt.ylabel("student numbers")
plt.title("SEIR model simulation\n (The chosen R0 value is {})".format(R0_))
plt.show()



f=open("Simulation.txt","w")
result3=scipy.integrate.odeint(SEIR_model,[S0,E0,I0,R0],t,args=(N,3.2*gamma,sigma,gamma))
print("Assuming default R0 value is 3.2 (SEIR model), after 15 weeks in the fall semester, the total amount of infected students will be ",int(1861-result3[14][0]),file=f)

## remove high centrality classes
classes_remove_5=["FT100/Lecture/1","FA102L/Lecture/1","BI242/Lecture/2","BI204/Lecture/2",
"CH201/Lecture/2"]
G1=G.copy()
G1.remove_nodes_from(classes_remove_5)
P1=bipartite.weighted_projected_graph(G1,student_index)
P_edges_weight=str(P.edges(data="weight"))
sum_weight_P=sum(int(i) for i in re.findall("(?='\d*', '\d*', (\d*))",P_edges_weight))
# print(sum_weight_P)
P1_edges_weight=str(P1.edges(data="weight"))
sum_weight_P1=sum(int(i) for i in re.findall("(?='\d*', '\d*', (\d*))",P1_edges_weight))
# print(sum_weight_P1)
print("\n\n\nAfter removing five classes with highest centrality, the expected value of R0 will decrease by"
,(1-sum_weight_P1/sum_weight_P)*100,"percentage","\nR0 value decreases from 3.2(default) to",3.2*sum_weight_P1/sum_weight_P,file=f)

result4_=scipy.integrate.odeint(SEIR_model,[S0,E0,I0,R0],t,args=(N,3.2*sum_weight_P1/sum_weight_P*gamma,sigma,gamma))
print("The total amount of infected students will decrease to",int(1861-result4_[14][0]),file=f)


classes_remove_10=["FT100/Lecture/1","FA102L/Lecture/1","BI242/Lecture/2","BI204/Lecture/2",
"CH201/Lecture/2","ID101/Lecture/2","BI323/Lecture/1","BI201/Lecture/1","BI329/Lecture/1","BI204/Lecture/1"]
G2=G.copy()
G2.remove_nodes_from(classes_remove_10)
P2=bipartite.weighted_projected_graph(G2,student_index)
P2_edges_weight=str(P2.edges(data="weight"))
sum_weight_P2=sum(int(i) for i in re.findall("(?='\d*', '\d*', (\d*))",P2_edges_weight))
print("\n\nAfter removing ten classes with highest centrality, the expected value of R0 will decrease by"
,(1-sum_weight_P2/sum_weight_P)*100,"percentage","\nR0 value decreases from 3.2(default) to",3.2*sum_weight_P2/sum_weight_P,file=f)

result4=scipy.integrate.odeint(SEIR_model,[S0,E0,I0,R0],t,args=(N,3.2*sum_weight_P2/sum_weight_P*gamma,sigma,gamma))
print("The total amount of infected students will decrease to",int(1861-result4[14][0]),file=f)



## the impact of the dining hall
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



dininghall_name=["dining_hall_M","dining_hall_T","dining_hall_W","dining_hall_R","dining_hall_F"]
G3=G.copy()
P3=P.copy()
G3.add_nodes_from(dininghall_name,bipartite=1)
student_dininghall=[]
for i in Lunch_M:
    if Lunch_M[i]=="Yes":
        student_dininghall.append((i,"dining_hall_M"))
for i in Lunch_T:
    if Lunch_T[i]=="Yes":
        student_dininghall.append((i,"dining_hall_T"))
for i in Lunch_W:
    if Lunch_W[i]=="Yes":
        student_dininghall.append((i,"dining_hall_W"))
for i in Lunch_R:
    if Lunch_R[i]=="Yes":
        student_dininghall.append((i,"dining_hall_R"))
for i in Lunch_F:
    if Lunch_F[i]=="Yes":
        student_dininghall.append((i,"dining_hall_F"))
# print(student_dininghall)


P_edges_weight=str(P.edges(data="weight"))
sum_weight_P=sum(int(i) for i in re.findall("(?='\d*', '\d*', (\d*))",P_edges_weight))

G3.add_edges_from(student_dininghall)
P3=bipartite.weighted_projected_graph(G3,student_index)
P3_edges_weight=str(P3.edges(data="weight"))
sum_weight_P3=sum(int(i) for i in re.findall("(?='\d*', '\d*', (\d*))",P3_edges_weight))
print("\n\nAfter opening the dining hall(assuming all possible students will have lunch in the dining hall),\nthe expected value of R0 will increase by"
,(sum_weight_P3/sum_weight_P-1)*100,"percentage","\nR0 value increases from 3.2(default) to",3.2*sum_weight_P3/sum_weight_P,file=f)
print("In this case, almost all the students will be infected",file=f)



# close classrooms with highest flow rate
Rooms_Dates={}
Rooms_Dates=defaultdict(list)
for i in file:
    for j in file[i]["Rooms"]:
        if j != "Unknown" and j != "Online Course, Online Course, Room Online" and j !="Online Course, Online Course, Room SYNC":
            if file[i]["Times"][file[i]["Rooms"].index(j)][0]!="Arranged":
                Rooms_Dates[j].append(file[i]["Times"][file[i]["Rooms"].index(j)][0])
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
# print(df_Rooms_Dates)

count_M=0
for i in df_Rooms_Dates["Monday"]:
    count_M += i*(i-1)/2
decrease_connection_M=max(df_Rooms_Dates["Monday"])*(max(df_Rooms_Dates["Monday"])-1)/2
count_T=0
for i in df_Rooms_Dates["Tuesday"]:
    count_T += i*(i-1)/2
decrease_connection_T=max(df_Rooms_Dates["Tuesday"])*(max(df_Rooms_Dates["Tuesday"])-1)/2
count_W=0
for i in df_Rooms_Dates["Wednesday"]:
    count_W += i*(i-1)/2
decrease_connection_W=max(df_Rooms_Dates["Wednesday"])*(max(df_Rooms_Dates["Wednesday"])-1)/2
count_R=0
for i in df_Rooms_Dates["Thursday"]:
    count_R += i*(i-1)/2
decrease_connection_R=max(df_Rooms_Dates["Thursday"])*(max(df_Rooms_Dates["Thursday"])-1)/2
count_F=0
for i in df_Rooms_Dates["Friday"]:
    count_F += i*(i-1)/2
decrease_connection_F=max(df_Rooms_Dates["Friday"])*(max(df_Rooms_Dates["Friday"])-1)/2
decrease_percentage=(decrease_connection_M+decrease_connection_T+decrease_connection_W+decrease_connection_R+decrease_connection_F)/(count_M+count_T+count_W+count_R+count_F)
print("\n\n\nAfter closing classrooms with highest flow rate each day(simply assuming students in the same room are connected each day),\nthe expected value of R0 will increase by"
,decrease_percentage*100,"percentage","\nR0 value decreases from 3.2(default) to",3.2*(1-decrease_percentage),file=f)

result5=scipy.integrate.odeint(SEIR_model,[S0,E0,I0,R0],t,args=(N,3.2*(1-decrease_percentage)*gamma,sigma,gamma))
print("The total amount of infected students will decrease to",int(1861-result5[14][0]),file=f)
