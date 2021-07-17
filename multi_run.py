# -*- coding: utf-8 -*-
"""
Created on Sat Jul 17 15:24:29 2021

@author: rapha
"""

import numpy as np
import matplotlib.pyplot as plt
import datetime

from model import *
from model import make_matrix_relations, make_relations_matrix_plot, make_money_list, HappyFolksAgent, HappyFolks

from tqdm import tqdm

n_iter=100

#%% Violence level test function
money_init="pre_defined"

def test_various_violence(list_of_list_violence,n_iter=20,nsteps_per_iter=50,matrix_init="pre_defined"):
    levels_results_happiness={}
    levels_results_money={}
    levels_results_couples={}
    levels_results_nagents={}
    levels_results_scenario={}
    levels_results_relations={}
    
    for violence_list in list_of_list_violence:
        print("Testing violence list %s"%(str(violence_list)))
        
        level_result_happiness=[]
        level_result_money=[]
        level_result_couples=[]
        level_result_nagents=[]
        level_result_scenario=[]
        level_result_relations=[]
        for i in tqdm(range(n_iter)):
            relations_matrix_stock=[]            
            couples_matrix_stock=[]
            money_list_stock=[]
            happiness_list_stock=[]
            n_alive_agents=[]
            scenario=[]
            
            
            model = HappyFolks(plotmat=False,plotmoney=False,plotstep=20,
                               matrix_init=matrix_init,money_init=money_init,verbose=0,
                               n_agent=len(names),external_disaster=False,violence_list=violence_list)
    
            for i in range(nsteps_per_iter):
                #print("Step",i)
                relations,couples,money,happiness,alive,latest_news=model.step()
                
                relations_matrix_stock.append(relations.copy())
                couples_matrix_stock.append(couples.copy())
                money_list_stock.append(money[:])
                happiness_list_stock.append(happiness[:])
                n_alive_agents.append(alive)
                scenario.append(latest_news)
            
            level_result_happiness.append(happiness_list_stock.copy())
            level_result_money.append(money_list_stock.copy())
            level_result_couples.append(list(map(lambda a: a[np.isnan(a)==False].sum()/2,couples_matrix_stock)).copy())
            level_result_nagents.append(n_alive_agents.copy())
            level_result_scenario.append(scenario.copy())
            level_result_relations.append(relations_matrix_stock.copy())
            
        levels_results_happiness[str(violence_list)]=level_result_happiness.copy()
        levels_results_money[str(violence_list)]=level_result_money.copy()
        levels_results_couples[str(violence_list)]=level_result_couples.copy()
        levels_results_nagents[str(violence_list)]=level_result_nagents.copy()
        levels_results_scenario[str(violence_list)]=level_result_scenario.copy()
        levels_results_relations[str(violence_list)]=level_result_relations.copy()

    return(levels_results_happiness,levels_results_money,levels_results_couples,levels_results_nagents,levels_results_relations)

#%% Various violence levels

happiness_dic,money_dic,couples_dic,nagents_dic,relations_dic=test_various_violence([[10,60,40,20,70,50],[10,20,10,10,10,20],[70,90,40,50,70,80],[90,10,20,20,30,20]],
                                                                                    n_iter=n_iter)

plt.figure(figsize=(10,10))
for key in list(happiness_dic.keys()):
    z=np.nansum(np.array(happiness_dic[key][:]),axis=2)
    t=np.array(nagents_dic[key][:])
    t=np.where(t==0,0.1,t) #avoid 0 division

    u=(z/t).mean(axis=0)
    plt.plot(u,label=key)

plt.legend()
plt.title('Average happiness of agents (alive), %s iter for various levels of violence_list'%(str(n_iter)))


plt.figure(figsize=(10,10))
for key in list(money_dic.keys()):
    z=np.nansum(np.array(money_dic[key][:]),axis=2)
    t=np.array(nagents_dic[key][:])
    t=np.where(t==0,0.1,t) #avoid 0 division
        
    u=(z/t).mean(axis=0)
    plt.plot(u,label=key)

plt.legend()
plt.title('Average money of agents (alive), %s iter for various levels of violence_list'%(str(n_iter)))


plt.figure(figsize=(10,10))
for key in list(couples_dic.keys()):
    z=np.array(couples_dic[key][:]).mean(axis=0)
    plt.plot(z,label=key)

plt.legend()
plt.title('Average number of couples, %s iter for various levels of violence_list'%(str(n_iter)))
    

plt.figure(figsize=(10,10))
for key in list(nagents_dic.keys()):
    z=np.array(nagents_dic[key][:]).mean(axis=0)
    plt.plot(z,label=key)

plt.legend()
plt.title('Average number of alive agents, %s iter for various levels of violence_list'%(str(n_iter)))
    

pos=1
plt.figure(figsize=(20,20))
plt.title("(pre_defined relations)")
for key in list(nagents_dic.keys()):
    for step in [0,19,39,49]:
        plt.subplot(4,4,pos)
        pos+=1
        plt.imshow(np.nanmean(relations_dic[key],axis=0)[step])
        plt.title("Violence parameters %s \n Episode %s"%(key,step+1))
        for i in range(len(names)):
            for j in range(len(names)):
                text = plt.text(j, i, round(np.nanmean(relations_dic[key],axis=0)[step][i,j],2),
                               ha="center", va="center", color="w")    

#%% Various violence levels with random relations initialization
money_init="pre_defined"

happiness_dic,money_dic,couples_dic,nagents_dic,relations_dic=test_various_violence([[10,60,40,20,70,50],[10,20,10,10,10,20],[70,90,40,50,70,80],[90,10,20,20,30,20]],
                                                                                    n_iter=n_iter,matrix_init="random")

plt.figure(figsize=(10,10))
for key in list(happiness_dic.keys()):
    z=np.nansum(np.array(happiness_dic[key][:]),axis=2)
    t=np.array(nagents_dic[key][:])
    t=np.where(t==0,0.1,t) #avoid 0 division

    u=(z/t).mean(axis=0)
    plt.plot(u,label=key)

plt.legend()
plt.title('Average happiness of agents (alive), %s iter for various levels of violence_list (random relations)'%(str(n_iter)))

plt.figure(figsize=(10,10))
for key in list(money_dic.keys()):
    z=np.nansum(np.array(money_dic[key][:]),axis=2)
    t=np.array(nagents_dic[key][:])
    t=np.where(t==0,0.1,t) #avoid 0 division
        
    u=(z/t).mean(axis=0)
    plt.plot(u,label=key)

plt.legend()
plt.title('Average money of agents (alive), %s iter for various levels of violence_list (random relations)'%(str(n_iter)))


plt.figure(figsize=(10,10))
for key in list(couples_dic.keys()):
    z=np.array(couples_dic[key][:]).mean(axis=0)
    plt.plot(z,label=key)

plt.legend()
plt.title('Average number of couples, %s iter for various levels of violence_list (random relations)'%(str(n_iter)))
    

plt.figure(figsize=(10,10))
for key in list(nagents_dic.keys()):
    z=np.array(nagents_dic[key][:]).mean(axis=0)
    plt.plot(z,label=key)

plt.legend()
plt.title('Average number of alive agents, %s iter for various levels of violence_list (random relations)'%(str(n_iter)))
    

pos=1
plt.figure(figsize=(20,20))
plt.title("(random relations)")
for key in list(nagents_dic.keys()):
    for step in [0,19,39,49]:
        plt.subplot(4,4,pos)
        pos+=1
        plt.imshow(np.nanmean(relations_dic[key],axis=0)[step])
        plt.title("Violence parameters %s \n Episode %s"%(key,step+1))
        for i in range(len(names)):
            for j in range(len(names)):
                text = plt.text(j, i, round(np.nanmean(relations_dic[key],axis=0)[step][i,j],2),
                               ha="center", va="center", color="w")    


#%% Single agent violence change

happiness_dic,money_dic,couples_dic,nagents_dic,relations_dic=test_various_violence([[10,60,40,20,70,50],[30,60,40,20,70,50],[50,60,40,20,70,50],[70,60,40,20,70,50],[90,60,40,20,70,50]],
                                                                                    matrix_init="random",n_iter=n_iter)

plt.figure(figsize=(10,10))
for key in list(happiness_dic.keys()):
    z=np.nanmean(np.array(happiness_dic[key][:])[:,:,0],axis=0)
    plt.plot(z,label=key)

plt.legend()
plt.title('Average happiness of Agent 0, %s iter for various levels of violence for Agent 0'%(str(n_iter)))



plt.figure(figsize=(10,10))
for key in list(happiness_dic.keys()):
    z=np.where(np.isnan(np.array(happiness_dic[key][:])[:,:,0]),0,1).mean(axis=0)
    plt.plot(z,label=key)

plt.legend()
plt.title('Probability of being alive of Agent 0, %s iter for various levels of violence for Agent 0'%(str(n_iter)))


#%% Sociability test function

money_init="pre_defined"

def test_various_sociability(list_of_list_sociability,n_iter=20,nsteps_per_iter=50,matrix_init="pre_defined"):
    levels_results_happiness={}
    levels_results_money={}
    levels_results_couples={}
    levels_results_nagents={}
    levels_results_scenario={}
    levels_results_relations={}
    
    for sociability_list in list_of_list_sociability:
        print("Testing sociability_list %s"%(str(sociability_list)))
        
        level_result_happiness=[]
        level_result_money=[]
        level_result_couples=[]
        level_result_nagents=[]
        level_result_scenario=[]
        level_result_relations=[]
        for i in tqdm(range(n_iter)):
            relations_matrix_stock=[]            
            couples_matrix_stock=[]
            money_list_stock=[]
            happiness_list_stock=[]
            n_alive_agents=[]
            scenario=[]
            
            
            model = HappyFolks(plotmat=False,plotmoney=False,plotstep=20,
                               matrix_init=matrix_init,money_init=money_init,verbose=0,
                               n_agent=len(names),external_disaster=False,sociability_list=sociability_list)
    
            for i in range(nsteps_per_iter):
                #print("Step",i)
                relations,couples,money,happiness,alive,latest_news=model.step()
                
                relations_matrix_stock.append(relations.copy())
                couples_matrix_stock.append(couples.copy())
                money_list_stock.append(money[:])
                happiness_list_stock.append(happiness[:])
                n_alive_agents.append(alive)
                scenario.append(latest_news)
            
            level_result_happiness.append(happiness_list_stock.copy())
            level_result_money.append(money_list_stock.copy())
            level_result_couples.append(list(map(lambda a: a[np.isnan(a)==False].sum()/2,couples_matrix_stock)).copy())
            level_result_nagents.append(n_alive_agents.copy())
            level_result_scenario.append(scenario.copy())
            level_result_relations.append(relations_matrix_stock.copy())
            
        levels_results_happiness[str(sociability_list)]=level_result_happiness.copy()
        levels_results_money[str(sociability_list)]=level_result_money.copy()
        levels_results_couples[str(sociability_list)]=level_result_couples.copy()
        levels_results_nagents[str(sociability_list)]=level_result_nagents.copy()
        levels_results_scenario[str(sociability_list)]=level_result_scenario.copy()
        levels_results_relations[str(sociability_list)]=level_result_relations.copy()

    return(levels_results_happiness,levels_results_money,levels_results_couples,levels_results_nagents,levels_results_relations)

#%% Various sociability levels
happiness_dic,money_dic,couples_dic,nagents_dic,relations_dic=test_various_sociability([[40,80,30,50,20,60],[90,70,80,60,90,60],[10,20,30,20,10,30],[10,80,30,90,10,20]],
                                                                                       n_iter=n_iter)

plt.figure(figsize=(10,10))
for key in list(happiness_dic.keys()):
    z=np.nansum(np.array(happiness_dic[key][:]),axis=2)
    t=np.array(nagents_dic[key][:])
    t=np.where(t==0,0.1,t) #avoid 0 division

    u=(z/t).mean(axis=0)
    plt.plot(u,label=key)

plt.legend()
plt.title('Average happiness of agents (alive), %s iter for various levels of sociability_list'%(str(n_iter)))

plt.figure(figsize=(10,10))
for key in list(money_dic.keys()):
    z=np.nansum(np.array(money_dic[key][:]),axis=2)
    t=np.array(nagents_dic[key][:])
    t=np.where(t==0,0.1,t) #avoid 0 division
        
    u=(z/t).mean(axis=0)
    plt.plot(u,label=key)

plt.legend()
plt.title('Average money of agents (alive), %s iter for various levels of sociability_list'%(str(n_iter)))


plt.figure(figsize=(10,10))
for key in list(couples_dic.keys()):
    z=np.array(couples_dic[key][:]).mean(axis=0)
    plt.plot(z,label=key)

plt.legend()
plt.title('Average number of couples, %s iter for various levels of sociability_list'%(str(n_iter)))
    

plt.figure(figsize=(10,10))
for key in list(nagents_dic.keys()):
    z=np.array(nagents_dic[key][:]).mean(axis=0)
    plt.plot(z,label=key)

plt.legend()
plt.title('Average number of alive agents, %s iter for various levels of sociability_list'%(str(n_iter)))
    

pos=1
plt.figure(figsize=(20,20))
plt.title("Title")
for key in list(nagents_dic.keys()):
    for step in [0,19,39,49]:
        plt.subplot(4,4,pos)
        pos+=1
        plt.imshow(np.nanmean(relations_dic[key],axis=0)[step])
        plt.title("Sociability parameters %s \n Episode %s"%(key,step+1))
        for i in range(len(names)):
            for j in range(len(names)):
                text = plt.text(j, i, round(np.nanmean(relations_dic[key],axis=0)[step][i,j],2),
                               ha="center", va="center", color="w")    

#%% Various sociability levels with random relations init 
happiness_dic,money_dic,couples_dic,nagents_dic,relations_dic=test_various_sociability([[40,80,30,50,20,60],[90,70,80,60,90,60],[10,20,30,20,10,30],[10,80,30,90,10,20]],
                                                                                       matrix_init="random",n_iter=n_iter)

plt.figure(figsize=(10,10))
for key in list(happiness_dic.keys()):
    z=np.nansum(np.array(happiness_dic[key][:]),axis=2)
    t=np.array(nagents_dic[key][:])
    t=np.where(t==0,0.1,t) #avoid 0 division

    u=(z/t).mean(axis=0)
    plt.plot(u,label=key)

plt.legend()
plt.title('Average happiness of agents (alive), %s iter for various levels of sociability_list  (random relations)'%(str(n_iter)))

plt.figure(figsize=(10,10))
for key in list(money_dic.keys()):
    z=np.nansum(np.array(money_dic[key][:]),axis=2)
    t=np.array(nagents_dic[key][:])
    t=np.where(t==0,0.1,t) #avoid 0 division
        
    u=(z/t).mean(axis=0)
    plt.plot(u,label=key)

plt.legend()
plt.title('Average money of agents (alive), %s iter for various levels of sociability_list  (random relations)'%(str(n_iter)))


plt.figure(figsize=(10,10))
for key in list(couples_dic.keys()):
    z=np.array(couples_dic[key][:]).mean(axis=0)
    plt.plot(z,label=key)

plt.legend()
plt.title('Average number of couples, %s iter for various levels of sociability_list  (random relations)'%(str(n_iter)))
    

plt.figure(figsize=(10,10))
for key in list(nagents_dic.keys()):
    z=np.array(nagents_dic[key][:]).mean(axis=0)
    plt.plot(z,label=key)

plt.legend()
plt.title('Average number of alive agents, %s iter for various levels of sociability_list  (random relations)'%(str(n_iter)))
    

pos=1
plt.figure(figsize=(20,20))
plt.title("Title")
for key in list(nagents_dic.keys()):
    for step in [0,19,39,49]:
        plt.subplot(4,4,pos)
        pos+=1
        plt.imshow(np.nanmean(relations_dic[key],axis=0)[step])
        plt.title("Sociability parameters %s \n Episode %s"%(key,step+1))
        for i in range(len(names)):
            for j in range(len(names)):
                text = plt.text(j, i, round(np.nanmean(relations_dic[key],axis=0)[step][i,j],2),
                               ha="center", va="center", color="w")    


#%% Single agent sociability change

happiness_dic,money_dic,couples_dic,nagents_dic,relations_dic=test_various_violence([[20,80,30,50,20,60],[40,80,30,50,20,60],[60,80,30,50,20,60],[80,80,30,50,20,60]],
                                                                                    matrix_init="random",n_iter=n_iter)

plt.figure(figsize=(10,10))
for key in list(happiness_dic.keys()):
    z=np.nanmean(np.array(happiness_dic[key][:])[:,:,0],axis=0)
    plt.plot(z,label=key)

plt.legend()
plt.title('Average happiness of Agent 0, %s iter for various levels of sociability for Agent 0'%(str(n_iter)))



plt.figure(figsize=(10,10))
for key in list(happiness_dic.keys()):
    z=np.where(np.isnan(np.array(happiness_dic[key][:])[:,:,0]),0,1).mean(axis=0)
    plt.plot(z,label=key)

plt.legend()
plt.title('Probability of being alive of Agent 0, %s iter for various levels of sociability for Agent 0'%(str(n_iter)))



#%% Exog events test function

money_init="pre_defined"

def test_various_exog(list_of_exog,n_iter=20,nsteps_per_iter=50,matrix_init="pre_defined"):
    levels_results_happiness={}
    levels_results_money={}
    levels_results_couples={}
    levels_results_nagents={}
    levels_results_scenario={}
    levels_results_relations={}
    
    for exog in list_of_exog:
        print("Testing Exogeneous events %s"%(str(exog)))
        
        level_result_happiness=[]
        level_result_money=[]
        level_result_couples=[]
        level_result_nagents=[]
        level_result_scenario=[]
        level_result_relations=[]
        for i in tqdm(range(n_iter)):
            relations_matrix_stock=[]            
            couples_matrix_stock=[]
            money_list_stock=[]
            happiness_list_stock=[]
            n_alive_agents=[]
            scenario=[]
            
            
            model = HappyFolks(plotmat=False,plotmoney=False,plotstep=20,
                               matrix_init=matrix_init,money_init=money_init,verbose=0,
                               n_agent=len(names),external_disaster=exog)
    
            for i in range(nsteps_per_iter):
                #print("Step",i)
                relations,couples,money,happiness,alive,latest_news=model.step()
                
                relations_matrix_stock.append(relations.copy())
                couples_matrix_stock.append(couples.copy())
                money_list_stock.append(money[:])
                happiness_list_stock.append(happiness[:])
                n_alive_agents.append(alive)
                scenario.append(latest_news)
            
            level_result_happiness.append(happiness_list_stock.copy())
            level_result_money.append(money_list_stock.copy())
            level_result_couples.append(list(map(lambda a: a[np.isnan(a)==False].sum()/2,couples_matrix_stock)).copy())
            level_result_nagents.append(n_alive_agents.copy())
            level_result_scenario.append(scenario.copy())
            level_result_relations.append(relations_matrix_stock.copy())
            
        levels_results_happiness[str(exog)]=level_result_happiness.copy()
        levels_results_money[str(exog)]=level_result_money.copy()
        levels_results_couples[str(exog)]=level_result_couples.copy()
        levels_results_nagents[str(exog)]=level_result_nagents.copy()
        levels_results_scenario[str(exog)]=level_result_scenario.copy()
        levels_results_relations[str(exog)]=level_result_relations.copy()

    return(levels_results_happiness,levels_results_money,levels_results_couples,levels_results_nagents,levels_results_relations)


#%% Various exog values
happiness_dic,money_dic,couples_dic,nagents_dic,relations_dic=test_various_exog([True,False],
                                                                                       n_iter=n_iter)

plt.figure(figsize=(10,10))
for key in list(happiness_dic.keys()):
    z=np.nansum(np.array(happiness_dic[key][:]),axis=2)
    t=np.array(nagents_dic[key][:])
    t=np.where(t==0,0.1,t) #avoid 0 division

    u=(z/t).mean(axis=0)
    plt.plot(u,label=key)

plt.legend()
plt.title('Average happiness of agents (alive), %s iter for various levels of exog'%(str(n_iter)))

plt.figure(figsize=(10,10))
for key in list(money_dic.keys()):
    z=np.nansum(np.array(money_dic[key][:]),axis=2)
    t=np.array(nagents_dic[key][:])
    t=np.where(t==0,0.1,t) #avoid 0 division
        
    u=(z/t).mean(axis=0)
    plt.plot(u,label=key)

plt.legend()
plt.title('Average money of agents (alive), %s iter for various levels of exog'%(str(n_iter)))


plt.figure(figsize=(10,10))
for key in list(couples_dic.keys()):
    z=np.array(couples_dic[key][:]).mean(axis=0)
    plt.plot(z,label=key)

plt.legend()
plt.title('Average number of couples, %s iter for various levels of exog'%(str(n_iter)))
    

plt.figure(figsize=(10,10))
for key in list(nagents_dic.keys()):
    z=np.array(nagents_dic[key][:]).mean(axis=0)
    plt.plot(z,label=key)

plt.legend()
plt.title('Average number of alive agents, %s iter for various levels of exog'%(str(n_iter)))
    

pos=1
plt.figure(figsize=(20,20))
plt.title("Title")
for key in list(nagents_dic.keys()):
    for step in [0,19,39,49]:
        plt.subplot(4,4,pos)
        pos+=1
        plt.imshow(np.nanmean(relations_dic[key],axis=0)[step])
        plt.title("exog parameters %s \n Episode %s"%(key,step+1))
        for i in range(len(names)):
            for j in range(len(names)):
                text = plt.text(j, i, round(np.nanmean(relations_dic[key],axis=0)[step][i,j],2),
                               ha="center", va="center", color="w")    

#%% Various sociability levels with random relations init 
happiness_dic,money_dic,couples_dic,nagents_dic,relations_dic=test_various_exog([True,False],
                                                                                       matrix_init="random",n_iter=n_iter)

plt.figure(figsize=(10,10))
for key in list(happiness_dic.keys()):
    z=np.nansum(np.array(happiness_dic[key][:]),axis=2)
    t=np.array(nagents_dic[key][:])
    t=np.where(t==0,0.1,t) #avoid 0 division

    u=(z/t).mean(axis=0)
    plt.plot(u,label=key)

plt.legend()
plt.title('Average happiness of agents (alive), %s iter for various levels of exog  (random relations)'%(str(n_iter)))

plt.figure(figsize=(10,10))
for key in list(money_dic.keys()):
    z=np.nansum(np.array(money_dic[key][:]),axis=2)
    t=np.array(nagents_dic[key][:])
    t=np.where(t==0,0.1,t) #avoid 0 division
        
    u=(z/t).mean(axis=0)
    plt.plot(u,label=key)

plt.legend()
plt.title('Average money of agents (alive), %s iter for various levels of exog  (random relations)'%(str(n_iter)))


plt.figure(figsize=(10,10))
for key in list(couples_dic.keys()):
    z=np.array(couples_dic[key][:]).mean(axis=0)
    plt.plot(z,label=key)

plt.legend()
plt.title('Average number of couples, %s iter for various levels of exog  (random relations)'%(str(n_iter)))
    

plt.figure(figsize=(10,10))
for key in list(nagents_dic.keys()):
    z=np.array(nagents_dic[key][:]).mean(axis=0)
    plt.plot(z,label=key)

plt.legend()
plt.title('Average number of alive agents, %s iter for various levels of exog  (random relations)'%(str(n_iter)))
    

pos=1
plt.figure(figsize=(20,20))
plt.title("Title")
for key in list(nagents_dic.keys()):
    for step in [0,19,39,49]:
        plt.subplot(4,4,pos)
        pos+=1
        plt.imshow(np.nanmean(relations_dic[key],axis=0)[step])
        plt.title("exog parameters %s \n Episode %s"%(key,step+1))
        for i in range(len(names)):
            for j in range(len(names)):
                text = plt.text(j, i, round(np.nanmean(relations_dic[key],axis=0)[step][i,j],2),
                               ha="center", va="center", color="w")    


#%% Couples type test function
money_init="pre_defined"

def test_various_genders(list_of_list_genders,n_iter=20,nsteps_per_iter=50,matrix_init="pre_defined"):
    levels_results_happiness={}
    levels_results_money={}
    levels_results_couples={}
    levels_results_nagents={}
    levels_results_scenario={}
    levels_results_relations={}
    
    for gender_list in list_of_list_genders:
        print("Testing gender list list %s"%(str(gender_list)))
        
        level_result_happiness=[]
        level_result_money=[]
        level_result_couples=[]
        level_result_nagents=[]
        level_result_scenario=[]
        level_result_relations=[]
        for i in tqdm(range(n_iter)):
            relations_matrix_stock=[]            
            couples_matrix_stock=[]
            money_list_stock=[]
            happiness_list_stock=[]
            n_alive_agents=[]
            scenario=[]
            
            
            model = HappyFolks(plotmat=False,plotmoney=False,plotstep=20,
                               matrix_init=matrix_init,money_init=money_init,verbose=0,
                               n_agent=len(names),external_disaster=False,gender_list=gender_list)
    
            for i in range(nsteps_per_iter):
                #print("Step",i)
                relations,couples,money,happiness,alive,latest_news=model.step()
                
                relations_matrix_stock.append(relations.copy())
                couples_matrix_stock.append(couples.copy())
                money_list_stock.append(money[:])
                happiness_list_stock.append(happiness[:])
                n_alive_agents.append(alive)
                scenario.append(latest_news)
            
            level_result_happiness.append(happiness_list_stock.copy())
            level_result_money.append(money_list_stock.copy())
            level_result_couples.append(list(map(lambda a: a[np.isnan(a)==False].sum()/2,couples_matrix_stock)).copy())
            level_result_nagents.append(n_alive_agents.copy())
            level_result_scenario.append(scenario.copy())
            level_result_relations.append(relations_matrix_stock.copy())
            
        levels_results_happiness[str(gender_list)]=level_result_happiness.copy()
        levels_results_money[str(gender_list)]=level_result_money.copy()
        levels_results_couples[str(gender_list)]=level_result_couples.copy()
        levels_results_nagents[str(gender_list)]=level_result_nagents.copy()
        levels_results_scenario[str(gender_list)]=level_result_scenario.copy()
        levels_results_relations[str(gender_list)]=level_result_relations.copy()

    return(levels_results_happiness,levels_results_money,levels_results_couples,levels_results_nagents,levels_results_relations)

#%% Various violence levels

happiness_dic,money_dic,couples_dic,nagents_dic,relations_dic=test_various_genders([[0,0,0,1,1,1],[0,0,0,0,0,0],[0,1,2,3,4,5]],
                                                                                    n_iter=n_iter)

plt.figure(figsize=(10,10))
for key in list(happiness_dic.keys()):
    z=np.nansum(np.array(happiness_dic[key][:]),axis=2)
    t=np.array(nagents_dic[key][:])
    t=np.where(t==0,0.1,t) #avoid 0 division

    u=(z/t).mean(axis=0)
    plt.plot(u,label=key)

plt.legend()
plt.title('Average happiness of agents (alive), %s iter for various levels of genders'%(str(n_iter)))


plt.figure(figsize=(10,10))
for key in list(money_dic.keys()):
    z=np.nansum(np.array(money_dic[key][:]),axis=2)
    t=np.array(nagents_dic[key][:])
    t=np.where(t==0,0.1,t) #avoid 0 division
        
    u=(z/t).mean(axis=0)
    plt.plot(u,label=key)

plt.legend()
plt.title('Average money of agents (alive), %s iter for various levels of genders'%(str(n_iter)))


plt.figure(figsize=(10,10))
for key in list(couples_dic.keys()):
    z=np.array(couples_dic[key][:]).mean(axis=0)
    plt.plot(z,label=key)

plt.legend()
plt.title('Average number of couples, %s iter for various levels of genders'%(str(n_iter)))
    

plt.figure(figsize=(10,10))
for key in list(nagents_dic.keys()):
    z=np.array(nagents_dic[key][:]).mean(axis=0)
    plt.plot(z,label=key)

plt.legend()
plt.title('Average number of alive agents, %s iter for various levels of genders'%(str(n_iter)))
    

pos=1
plt.figure(figsize=(20,20))
plt.title("(pre_defined relations)")
for key in list(nagents_dic.keys()):
    for step in [0,19,39,49]:
        plt.subplot(4,4,pos)
        pos+=1
        plt.imshow(np.nanmean(relations_dic[key],axis=0)[step])
        plt.title("genders parameters %s \n Episode %s"%(key,step+1))
        for i in range(len(names)):
            for j in range(len(names)):
                text = plt.text(j, i, round(np.nanmean(relations_dic[key],axis=0)[step][i,j],2),
                               ha="center", va="center", color="w")    


#%% Various violence levels with random relations initialization
money_init="pre_defined"

happiness_dic,money_dic,couples_dic,nagents_dic,relations_dic=test_various_genders([[0,0,0,1,1,1],[0,0,0,0,0,0],[0,1,2,3,4,5]],
                                                                                    n_iter=n_iter,matrix_init="random")

plt.figure(figsize=(10,10))
for key in list(happiness_dic.keys()):
    z=np.nansum(np.array(happiness_dic[key][:]),axis=2)
    t=np.array(nagents_dic[key][:])
    t=np.where(t==0,0.1,t) #avoid 0 division

    u=(z/t).mean(axis=0)
    plt.plot(u,label=key)

plt.legend()
plt.title('Average happiness of agents (alive), %s iter for various levels of genders (random relations)'%(str(n_iter)))

plt.figure(figsize=(10,10))
for key in list(money_dic.keys()):
    z=np.nansum(np.array(money_dic[key][:]),axis=2)
    t=np.array(nagents_dic[key][:])
    t=np.where(t==0,0.1,t) #avoid 0 division
        
    u=(z/t).mean(axis=0)
    plt.plot(u,label=key)

plt.legend()
plt.title('Average money of agents (alive), %s iter for various levels of genders (random relations)'%(str(n_iter)))


plt.figure(figsize=(10,10))
for key in list(couples_dic.keys()):
    z=np.array(couples_dic[key][:]).mean(axis=0)
    plt.plot(z,label=key)

plt.legend()
plt.title('Average number of couples, %s iter for various levels of genders (random relations)'%(str(n_iter)))
    

plt.figure(figsize=(10,10))
for key in list(nagents_dic.keys()):
    z=np.array(nagents_dic[key][:]).mean(axis=0)
    plt.plot(z,label=key)

plt.legend()
plt.title('Average number of alive agents, %s iter for various levels of genders (random relations)'%(str(n_iter)))
    

pos=1
plt.figure(figsize=(20,20))
plt.title("(random relations)")
for key in list(nagents_dic.keys()):
    for step in [0,19,39,49]:
        plt.subplot(4,4,pos)
        pos+=1
        plt.imshow(np.nanmean(relations_dic[key],axis=0)[step])
        plt.title("genders parameters %s \n Episode %s"%(key,step+1))
        for i in range(len(names)):
            for j in range(len(names)):
                text = plt.text(j, i, round(np.nanmean(relations_dic[key],axis=0)[step][i,j],2),
                               ha="center", va="center", color="w")    
