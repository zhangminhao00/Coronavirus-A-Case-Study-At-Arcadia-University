import json
import networkx as nx
from networkx.algorithms import bipartite
import matplotlib.pyplot as plt
import operator
import re
import numpy as np
import pandas as pd
from pandas import Series,DataFrame
from collections import defaultdict
import scipy.integrate
import seaborn as sns

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

def SEIR_model(y,t,N,beta,sigma,gamma):
    S,E,I,R= y
    dS_dt= -beta*S*I/N
    dE_dt= beta*S*I/N-sigma*E
    dI_dt= sigma*E-gamma*I
    dR_dt= gamma*I
    return([dS_dt,dE_dt,dI_dt,dR_dt])

t= np.linspace(0,80,80)


## Sensitivity Analysis
value_afterchange=[]
initial_value=[]
parameters_name=[]
rate_change=[]
results=[]
results_change=[]
amount_before=[]
difference_rate=[]
for i in range(0,5):
    for change_rate in [-0.3,-0.1,0.1,0.3]:
        N=1861
        E0=0
        I0=1861*688/100000
        R0=0
        S0=N-I0-E0-R0
        sigma=1/5.2*7
        gamma=1/7
        beta=3.2/7
        if i==0:
            I0=I0*(1+change_rate)
            value_afterchange.append(I0)
        if i==1:
            N=N*(1+change_rate)
            value_afterchange.append(N)
        if i==2:
            beta=beta*(1+change_rate)
            value_afterchange.append(beta)
        if i==3:
            sigma=sigma*(1+change_rate)
            value_afterchange.append(sigma)
        if i==4:
            gamma=gamma*(1+change_rate)
            value_afterchange.append(gamma)
        susceptible_final=scipy.integrate.odeint(SEIR_model,[S0,E0,I0,R0],t,args=(N,beta,sigma,gamma))
        affected_students=int(N-susceptible_final[14][0])
        results.append(affected_students)
        results_change.append(affected_students-456)
for i in range(0,5):
    for j in ["-30%","-10%","+10%","+30%"]:
        rate_change.append(str(j))
for i in[1861*688/100000,1861,3.2/7,1/5.2*7,1/7]:
    for j in range(0,4):
        initial_value.append(i)
for i in ["Infectious_initial(I0)","number of students(N)","beta","sigma","gamma"]:
    for j in range(0,4):
        parameters_name.append(i)
for i in range(0,20):
    amount_before.append(456)
for i in range(0,20):
    difference_rate.append(results_change[i]/456)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
data={"parameters":parameters_name,"initial value":initial_value,"change rate":rate_change,
"value after change":value_afterchange,"the number of affected students after change":results,
"the number of affected students before change":amount_before,
"difference value":results_change,"difference rate":difference_rate}
df= DataFrame(data)
print(df.to_string(justify='center', index=False))

writer=pd.ExcelWriter("Sensitivity_Analysis(R0=3.2).xlsx")
df.to_excel(writer,sheet_name="Sensitivity Analysis")
writer.save()
