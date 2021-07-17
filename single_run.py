# -*- coding: utf-8 -*-
"""
Created on Sat Jul 17 15:21:54 2021

@author: rapha
"""

import numpy as np
import matplotlib.pyplot as plt
import datetime

from model import *
from model import make_matrix_relations, make_relations_matrix_plot, make_money_list, HappyFolksAgent, HappyFolks


#%% Parameters 

#In model


#%% Stock lists
relations_matrix_stock=[]
couples_matrix_stock=[]
money_list_stock=[]
happiness_list_stock=[]
n_alive_agents=[]
scenario=[]

#%% init
money_init="pre_defined"
matrix_init="pre_defined"
nsteps=100

model = HappyFolks(plotmat=False,plotmoney=False,plotstep=20,
                   matrix_init=matrix_init,money_init=money_init,verbose=0,
                   n_agent=len(names),external_disaster=False)

#%% Run
for i in range(nsteps):
    print("Step",i)
    relations,couples,money,happiness,alive,latest_news=model.step()
    
    relations_matrix_stock.append(relations.copy())
    couples_matrix_stock.append(couples.copy())
    money_list_stock.append(money[:])
    happiness_list_stock.append(happiness[:])
    n_alive_agents.append(alive)
    scenario.append(latest_news)



#%% Plot figures

plt.figure(figsize=(10,10))
labels=names    

for y_arr, label in zip(np.transpose(np.array(money_list_stock)), labels):
    if np.nanmax(np.transpose(np.array(money_list_stock)))>10000:
        plt.plot(range(nsteps), np.log(y_arr), label=label+" (log)")
    else:
        plt.plot(range(nsteps), y_arr, label=label)

if nsteps<=20:   
    plt.xticks(range(nsteps),range(1,nsteps+1))   
    plt.vlines(range(nsteps),ymin=np.array(money_list_stock).min(),ymax=np.array(money_list_stock).max(),colors="black",linestyles="dashed")
plt.legend()
plt.title('Money')
plt.show()



plt.figure(figsize=(10,10))
labels=names
for y_arr, label in zip(np.transpose(np.array(happiness_list_stock)), labels):
    plt.plot(range(nsteps), y_arr, label=label)
    
if nsteps<=20:   
    plt.xticks(range(nsteps),range(1,nsteps+1))
    plt.vlines(range(nsteps),ymin=np.array(happiness_list_stock).min(),ymax=np.array(happiness_list_stock).max(),colors="black",linestyles="dashed")
plt.legend()
plt.title('Happiness')
plt.show()


plt.figure(figsize=(10,10))
fig,ax = plt.subplots(figsize=(10,10))
ax.plot(n_alive_agents,color="red")
ax.set_xlabel("step",fontsize=14)
ax.set_ylabel("number of agents alive",color="red")

if nsteps<=20:
    ax.set_xticks(range(nsteps))
    ax.set_xticklabels(range(1,nsteps+1))
#ax2=ax.twinx()
#ax2.plot(np.array(happiness_list_stock).sum(axis=1),color="blue")
#ax2.set_ylabel("total happiness",color="blue")
ax3=ax.twinx()
ax3.plot(np.nansum(np.array(happiness_list_stock),axis=1)/n_alive_agents,color="orange")
ax3.set_ylabel("average happiness",color="orange")

#%% Write the scenario
path=r"C:\Users\rapha\Desktop\Markov Chains & Agent Based Systems\Project\example_happyfolks\scenarios"

now=datetime.datetime.now().strftime("%d_%b_%Y_%H_%M_%S")

with open(path+'\\scenario_'+now+'.txt', 'w') as f:
    f.write("Scenario - Telenabehem\n")
    f.write(datetime.datetime.now().strftime("%d-%b-%Y (%H:%M)"))
    f.write("\n")
    f.write('\n ------------------------------------------------------------- \n')
    for i,name in enumerate(names):
        f.write("------ Name %s ------ \n"%(name))
        f.write("Violence level : %s \n"%(model.violence_list[i]))
        f.write("Risk taker (business) : %s \n"%(model.business_risk[i]))
        f.write("Sociability : %s \n"%(model.sociability_list[i]))
        if money_init=="pre_defined":
            f.write("Money at start : %s \n"%(make_money_list(names)[i]))
            
        if matrix_init=="pre_defined":
            matrix=make_matrix_relations(names,mode="pre_defined")
            matrix_line=matrix[i,:]
            for j in range(len(matrix_line)):
                if (j!=i):
                    f.write(" Relation with %s : %s \n"%(names[int(j)],matrix_line[int(j)]))
        
    f.write('\n ------------------------------------------------------------- \n')    
    
    for line in scenario:
        f.write(line)
        f.write('\n ------------------------------------------------------------- \n')
