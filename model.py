#Happy Folks

#%% Initialization
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd 
import gc
import scipy.stats as stats
import math
import datetime
from tqdm import tqdm
import sys

path=r'C:\Users\rapha\Desktop\Markov Chains & Agent Based Systems\Project\example'
sys.path.insert(1, path)

from mesa import Model, Agent
from mesa.time import RandomActivation
from mesa.space import SingleGrid
from mesa.datacollection import DataCollector

#%% Parameters


names=["Bill","Bob","Brad","Mary","Margaret","Melany"]

# violence_list=[10,60,40,20,70,50]
# business_risk=[0.2,0.1,0.4,0.2,0.5,0.1]
# gender_list=[1,1,1,0,0,0]
# happiness_list=[0,0,0,0,0,0]
# sociability_list=[40,80,30,50,20,60]

#violence
violent_relation_thres=0.4
violence_thres = 50
violence_death_thres=0.2

#couples
couples_thres=0.9
couples_random_thres=1

#business coop
business_coop_thres = 0.6
business_coop_randm_thres=1

#conspiracy
conspiracy_thres=0.5
conspiracy_random_thres=0.9
ennemy_thres=0.3
high_violence_thres=60

#suicide
suicide_thres_happiness=-10
suicide_thres_money=3
suicide_random_thres=0.25

#%% Create relations matrix

def make_matrix_relations(names,mode="user_defined"):
    matrix=np.zeros((len(names),len(names)))
    if mode=="user_defined":
        for i in range(len(names)):
            for j in range(len(names)):
                if i<j:
                    r=input("Relation between %s and %s : \n"%(names[i],names[j]))
                    matrix[i,j]=r
                    matrix[j,i]=r  
    elif mode=="random":
        for i in range(len(names)):
            for j in range(len(names)):
                if i<j:
                    r=random.random()
                    matrix[i,j]=r       
                    matrix[j,i]=r  
    else:
        matrix=np.array([[0. , 0.9, 0.4, 0.5, 0.8, 0.3],
                           [0.9, 0. , 0.7, 0.5, 0.8, 0.7],
                           [0.4, 0.7, 0. , 0.8, 0.3, 0.7],
                           [0.5, 0.5, 0.8, 0. , 0.3, 0.9],
                           [0.8, 0.8, 0.3, 0.3, 0. , 0.4],
                           [0.3, 0.7, 0.7, 0.9, 0.4, 0. ]])        

    return(matrix)

#%% Create relations matrix plot
 
def make_relations_matrix_plot(relations_matrix,title="Relations"):
    
    fig, ax = plt.subplots(1,1)
    img = ax.matshow(relations_matrix,vmin=0,vmax=1)
    
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names,rotation=90)
    
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_title(title)

    for i in range(len(names)):
        for j in range(len(names)):
            text = ax.text(j, i, round(relations_matrix[i, j],2),
                           ha="center", va="center", color="w")
    fig.colorbar(img)
    
    return(fig)


#%% Create money list

def make_money_list(names,mode="pre_defined"):
    if mode=="pre_defined":
        return([100, 100, 100, 100, 100, 100])
    elif mode=="used_defined":
        mlist=[]
        for i in names:
            money=input("Money level for %s"%(i))
            mlist.append(money)
        return(mlist)
    else:
        return(np.random.uniform(10,1000,size=len(names)).tolist())

#%% Model

class HappyFolksAgent(Agent):
    """
    Happy Folks agent
    """

    def __init__(self, pos, model, agent_type, agent_id):
        """
        Create a new HF agent.

        Args:
           unique_id: Unique identifier for the agent.
           x, y: Agent initial location.
           agent_type: Indicator for the agent's type (minority=1, majority=0)
        """
        super().__init__(pos, model)
        self.pos = pos
        self.agent_id=agent_id
        self.agent_name=names[self.agent_id]
        self.type = agent_type
        self.happiness = 0
        self.violence = self.model.violence_list[self.agent_id]
        self.gender = self.model.gender_list[self.agent_id]
        self.business_risk = self.model.business_risk[self.agent_id]
        self.sociability = self.model.sociability_list[self.agent_id]
        self.money = self.model.money_list[self.agent_id]
        self.dead=0
        
    def add_to_latest_news(self,text):
        if self.model.verbose>=1:
            print(text)
            
        text="\n --"+text
        
        self.model.latest_news+=text

    def step(self):
        if self.dead==0:
            if self.model.verbose>=1:
                print(self.agent_name,"Happiness",self.happiness,"Money",self.money)
            
            self.model.alive_agents+=1
            neighbor_num=0
            
            if self.model.external_disaster:
                if random.random()<self.model.earthquake_proba/self.model.n_agent:
                    self.add_to_latest_news("*** An earthquake has hit the city ***")
                    visited=[]                                    
                    for obj in gc.get_objects():
                        if isinstance(obj, HappyFolksAgent):
                            idagent=obj.agent_id
                            if (idagent not in visited):
                                visited.append(obj.agent_id)
                                if (random.random()<0.05):
                                    self.add_to_latest_news("%s died in the earthquake"%(obj.agent_name))
                                    visited_bis=[]                                    
                                    for other_obj in gc.get_objects():
                                        if isinstance(other_obj, HappyFolksAgent):
                                            idotheragent=other_obj.agent_id
                                            if (idotheragent not in visited_bis):
                                                visited_bis.append(other_obj.agent_id)
                                                if (self.model.relations_matrix[obj.agent_id,idotheragent]>=0.5):
                                                    self.add_to_latest_news("%s is sad after %s death"%(other_obj.agent_name,obj.agent_name))
                                                    other_obj.happiness+=-2
                                            
                                    obj.dead=1                       
                                    
                                    self.model.couples_matrix[obj.agent_id,:]=np.nan
                                    self.model.couples_matrix[:,obj.agent_id]=np.nan
                                    self.model.relations_matrix[obj.agent_id,:]=np.nan
                                    self.model.relations_matrix[:,obj.agent_id]=np.nan 
                                    
                                    obj.money=-1
                                    self.model.money_list[obj.agent_id]=-1                                                    
                    
                                elif (random.random()<0.3):
                                    self.add_to_latest_news("%s had its house destroyed in the earthquake"%(obj.agent_name))
                                    obj.happiness+=-4
                                else:
                                    obj.happiness+=-2
                                    
                elif random.random()<self.model.financialcrisis_proba/self.model.n_agent:
                    loss=random.uniform(0.5,1)
                    self.add_to_latest_news("*** A financial crisis struck and the agents lost %s percent of their money"%(loss))
                    visited=[]                                    
                    for obj in gc.get_objects():
                        if isinstance(obj, HappyFolksAgent):
                            idagent=obj.agent_id
                            if (idagent not in visited):
                                visited.append(obj.agent_id)
                                crisis_opportunity=random.uniform(obj.business_risk,1)                                            
                                if (crisis_opportunity>0.9):
                                    self.add_to_latest_news("%s made good choices and benefitted from the crisis"%(obj.agent_name))
                                    obj.happiness+=2
                                    obj.money*=2
                                    self.model.money_list[obj.agent_id]*=2
                                    
                                else:
                                    obj.happiness+=-2
                                    obj.money*=(1-loss)   
                                    self.model.money_list[obj.agent_id]*=(1-loss)

            
            for neighbor in self.model.grid.neighbor_iter(self.pos):
                if neighbor.dead==0:
                    if neighbor_num<1:
                        relation_quality=self.model.relations_matrix[self.agent_id,neighbor.agent_id]
                        
                        if relation_quality>0.5:
                            happiness_var=self.model.relation_like_happiness_var
                            if self.sociability > 70:
                                happiness_var+=2
                                self.add_to_latest_news("%s met with %s and is thrilled about it"%(self.agent_name,neighbor.agent_name))
                            elif self.sociability > 40:
                                happiness_var+=1
                                self.add_to_latest_news("%s met with %s and is happy about it"%(self.agent_name,neighbor.agent_name))
                            else:
                                self.add_to_latest_news("%s met with %s and is fine but does not really care about other people"%(self.agent_name,neighbor.agent_name))

                        else:
                            happiness_var=self.model.relation_dislike_happiness_var
                            self.add_to_latest_news("%s met with %s and does not like it"%(self.agent_name,neighbor.agent_name))
                            
                        ##Fighting
                        if relation_quality<violent_relation_thres:
                            if (self.violence>violence_thres) or (neighbor.violence>violence_thres):
                                if random.random()<0.3:
                                    self.add_to_latest_news("%s and %s fought"%(self.agent_name,neighbor.agent_name))
                                    self.model.relations_matrix[self.agent_id,neighbor.agent_id]+=-0.3
                                    self.model.relations_matrix[neighbor.agent_id,self.agent_id]+=-0.3
                                    
                                    if random.randint(0,self.violence+neighbor.violence)<self.violence:
                                        happiness_var+=2
                                        neighbor.happiness+=-3
                                        self.add_to_latest_news("%s threw %s on the ground"%(self.agent_name, neighbor.agent_name))
                                        visited=[]
                                        for obj in gc.get_objects():
                                            if isinstance(obj, HappyFolksAgent):
                                                idagent=obj.agent_id
                                                if (idagent not in visited):
                                                    visited.append(obj.agent_id)
                                                    if (self.model.relations_matrix[neighbor.agent_id,idagent]>=0.6):
                                                        self.add_to_latest_news("%s is sad for %s"%(obj.agent_name,neighbor.agent_name))
                                                        obj.happiness+=-1
                                                    
                                        if random.random()<violence_death_thres:
                                            self.add_to_latest_news("%s has been killed in the fight"%(neighbor.agent_name))
                                            neighbor.dead=1                                                   
                                            self.model.couples_matrix[neighbor.agent_id,:]=np.nan
                                            self.model.couples_matrix[:,neighbor.agent_id]=np.nan
                                            self.model.relations_matrix[neighbor.agent_id,:]=np.nan
                                            self.model.relations_matrix[:,neighbor.agent_id]=np.nan 
                                            
                                            neighbor.money=-1
                            
                                    else:
                                        happiness_var+=-3
                                        neighbor.happiness+=2
                                        self.add_to_latest_news("%s threw %s on the ground"%(neighbor.agent_name,self.agent_name))
                                        visited=[]                                    
                                        for obj in gc.get_objects():
                                            if isinstance(obj, HappyFolksAgent):
                                                idagent=obj.agent_id
                                                if  (idagent not in visited):
                                                    visited.append(obj.agent_id)                                            
                                                    if (self.model.relations_matrix[neighbor.agent_id,idagent]>=0.6):
                                                        self.add_to_latest_news("%s is sad for %s"%(obj.agent_name,self.agent_name))
                                                        obj.happiness+=-1
                                                    
                                        if random.random()<violence_death_thres:
                                            self.add_to_latest_news("%s has been killed in the fight"%(self.agent_name))
                                            self.dead=1                                                   
                                            self.model.couples_matrix[self.agent_id,:]=np.nan
                                            self.model.couples_matrix[:,self.agent_id]=np.nan
                                            self.model.relations_matrix[self.agent_id,:]=np.nan
                                            self.model.relations_matrix[:,self.agent_id]=np.nan 
                                            
                                            self.money=-1
                                                
                        ##Business common
                        if (relation_quality>business_coop_thres)&(random.random()<business_coop_randm_thres):
                            
                            std=(self.business_risk+neighbor.business_risk)/2
                            sucess=np.random.normal(0, std, 1)[0]
                            self.money*=(1+sucess)
                            self.model.money_list[self.agent_id]*=(1+sucess)                    
                            neighbor.money*=(1+sucess)
                            self.model.money_list[neighbor.agent_id]*=(1+sucess)
        
                            if sucess>0:
                                happiness_var+=2
                                neighbor.happiness=neighbor.happiness+2
                                self.model.relations_matrix[self.agent_id,neighbor.agent_id]+=0.2
                                self.model.relations_matrix[neighbor.agent_id,self.agent_id]+=0.2                        
                            else:
                                happiness_var+=-2          
                                neighbor.happiness=neighbor.happiness-2
                            
                            self.add_to_latest_news("%s and %s made business and made %2.f percent out of it"%(self.agent_name,neighbor.agent_name,sucess*100))
                            
                        ##Couple/Breakup
                        
                        if (relation_quality>couples_thres) and (random.random()<couples_random_thres):
                            if (self.gender!=neighbor.gender) and (self.model.couples_matrix[self.agent_id,neighbor.agent_id]!=1):
                                self.add_to_latest_news("I love you my dear %s, yours for ever %s"%(neighbor.agent_name,self.agent_name))
                                happiness_var+=3
                                
                                #if the agent was previously engaged
                                if (self.model.couples_matrix[self.agent_id,:]>0).any():
                                    previous_partner_id=np.where(self.model.couples_matrix[self.agent_id,:]>0)[0][0]
                                    visited=[]
                                    for obj in gc.get_objects():
                                        if isinstance(obj, HappyFolksAgent):
                                            idagent=obj.agent_id
                                            if (idagent not in visited):
                                                visited.append(idagent)
                                                if (idagent==previous_partner_id):
                                                    self.add_to_latest_news("%s is mad at %s for breaking their couple apart"%(obj.agent_name,self.agent_name))
                                                    obj.happiness+=-4
                                                    self.model.relations_matrix[self.agent_id,idagent]+=-0.4
                                                    self.model.relations_matrix[idagent,self.agent_id]+=-0.4
                                                    self.model.couples_matrix[self.agent_id,idagent]=0
                                                    self.model.couples_matrix[idagent,self.agent_id]=0
                                                
                                #if the neighbor was previously engaged
                                if (self.model.couples_matrix[neighbor.agent_id,:]>0).any():
                                    previous_partner_id=np.where(self.model.couples_matrix[neighbor.agent_id,:]>0)[0][0]
                                    visited=[]
                                    for obj in gc.get_objects():
                                        if isinstance(obj, HappyFolksAgent):
                                            idagent=obj.agent_id
                                            if (idagent not in visited):
                                                visited.append(idagent)
                                                if idagent==previous_partner_id:
                                                    self.add_to_latest_news("%s is mad at %s for breaking their couple apart"%(obj.agent_name,neighbor.agent_name))
                                                    obj.happiness+=-4
                                                    self.model.relations_matrix[neighbor.agent_id,idagent]+=-0.4
                                                    self.model.relations_matrix[idagent,neighbor.agent_id]+=-0.4
                                                    self.model.couples_matrix[neighbor.agent_id,idagent]=0
                                                    self.model.couples_matrix[idagent,neighbor.agent_id]=0
                                    
                                                    
                                self.model.couples_matrix[self.agent_id,neighbor.agent_id]=1
                                self.model.couples_matrix[neighbor.agent_id,self.agent_id]=1
        
                            elif self.model.couples_matrix[self.agent_id,neighbor.agent_id]==1:
                                self.add_to_latest_news("Glad to see you my beloved %s, yours lovely %s"%(neighbor.agent_name,self.agent_name))
                                happiness_var+=1
                                
                        ##Common ennemy
                        if (relation_quality>conspiracy_thres) and (random.random()<conspiracy_random_thres):
                            common_ennemy=False
                            for obj in gc.get_objects():
                                if isinstance(obj, HappyFolksAgent):
                                    idagent=obj.agent_id
                                    if (self.model.relations_matrix[self.agent_id,idagent]<ennemy_thres)&(self.model.relations_matrix[neighbor.agent_id,idagent]<ennemy_thres):
                                        self.add_to_latest_news("%s and %s have %s as common ennemy and they will break him/her !"%(self.agent_name,neighbor.agent_name,obj.agent_name))
                                        common_ennemy=True
                                        break
                                    
                            if common_ennemy:        
                                if ((self.violence+neighbor.violence)/2>high_violence_thres) and (random.random()<0.9):
                                    self.add_to_latest_news("They killed %s"%(obj.agent_name))
                                    
                                    obj.dead=1                       
                                
                                    self.model.couples_matrix[obj.agent_id,:]=np.nan
                                    self.model.couples_matrix[:,obj.agent_id]=np.nan
                                    self.model.relations_matrix[obj.agent_id,:]=np.nan
                                    self.model.relations_matrix[:,obj.agent_id]=np.nan 
                                    
                                    obj.money=-1
                                
                                elif random.random()<0.5:
                                    self.add_to_latest_news("They ruined %s"%(obj.agent_name))
                                    obj.happiness=obj.happiness-2
                                    
                                    rate=random.uniform(0.3,1)
                                    obj.money=obj.money*(1-rate)
                                    self.model.money_list[obj.agent_id]*=(1-rate)
                                else:
                                    if self.model.couples_matrix[obj.agent_id,:].sum()>0:
                                        partner_id=np.where(self.model.couples_matrix[obj.agent_id,:]>0)[0][0]
                                        self.model.couples_matrix[obj.agent_id,partner_id]=0 
                                        self.model.couples_matrix[partner_id,obj.agent_id]=0 
                                        self.model.relations_matrix[obj.agent_id,partner_id]+=-0.5
                                        self.model.relations_matrix[partner_id,obj.agent_id]+=-0.5                              
                                        obj.happiness=obj.happiness-2
                                                    
                                        self.add_to_latest_news("They ruined %s couple with %s"%(obj.agent_name,names[partner_id]))
                                        
                                    else:
                                        obj.happiness=obj.happiness-2
                                        self.add_to_latest_news("They affected %s happiness"%(obj.agent_name))
                                                                                                                   
                        ###
                        self.add_to_latest_news("Hapiness var of %s : %2.f"%(self.agent_name,happiness_var))
                        self.happiness += happiness_var
                        self.model.model_happiness += happiness_var
                        self.model.happiness_list[self.agent_id]=self.happiness
    
                                            
                        
                                
                        
                        random_rel_var=np.random.uniform(-0.1,0.2)
                        self.model.relations_matrix[self.agent_id,neighbor.agent_id]+=random_rel_var
                        self.model.relations_matrix[neighbor.agent_id,self.agent_id]+=random_rel_var
        
                        self.model.relations_matrix[self.agent_id,neighbor.agent_id]=max(min(self.model.relations_matrix[self.agent_id,neighbor.agent_id],1),0)
                        self.model.relations_matrix[neighbor.agent_id,self.agent_id]=max(min(self.model.relations_matrix[neighbor.agent_id,self.agent_id],1),0)
                        
                neighbor_num+=1
                
            
            if neighbor_num==0:
                if self.sociability > 70 and random.random()<0.4:
                    self.happiness+=-2
                elif self.sociability > 40 and random.random()<0.2:
                    self.happiness+=-2
                elif self.sociability <= 20 and random.random()<0.2 :
                    self.happiness+=1
                    
                if (self.happiness<suicide_thres_happiness) or (self.money<suicide_thres_money):
                    if (len(np.where(self.model.couples_matrix[self.agent_id,:]>0)[0])==0):
                        if (random.random()<suicide_random_thres):
                            
                            if (self.money<suicide_thres_money):
                                self.add_to_latest_news("=== %s suicided out of financial ruin === "%(self.agent_name))
                            else:
                                self.add_to_latest_news("=== %s suicided out of sadness ==="%(self.agent_name))

                            visited=[]                                    
                            for obj in gc.get_objects():
                                if isinstance(obj, HappyFolksAgent):
                                    idagent=obj.agent_id
                                    if (idagent not in visited):
                                        visited.append(obj.agent_id)
                                        if (self.model.relations_matrix[self.agent_id,idagent]>=0.5):
                                            self.add_to_latest_news("%s is sad after %s suicide"%(obj.agent_name,self.agent_name))
                                            obj.happiness+=-2
                                            
                            self.dead=1                       
                            
                            self.model.couples_matrix[self.agent_id,:]=np.nan
                            self.model.couples_matrix[:,self.agent_id]=np.nan
                            self.model.relations_matrix[self.agent_id,:]=np.nan
                            self.model.relations_matrix[:,self.agent_id]=np.nan 
                            
                            self.money=-1
                            
                    else:
                        if (random.random()<suicide_random_thres/2):
                            if (self.money<suicide_thres_money):
                                self.add_to_latest_news("=== %s suicided out of financial ruin === "%(self.agent_name))
                            else:
                                self.add_to_latest_news("=== %s suicided out of sadness ==="%(self.agent_name))
                            partner_id=np.where(self.model.couples_matrix[self.agent_id,:]>0)[0][0]
                            visited=[]
                            for obj in gc.get_objects():
                                if isinstance(obj, HappyFolksAgent):
                                    idagent=obj.agent_id
                                    if (idagent not in visited):
                                        visited.append(idagent)
                                        if idagent==partner_id:
                                            partner=obj
                                            self.add_to_latest_news("=== %s sucided with %s out of love ==="%(obj.agent_name,self.agent_name))


                            visited=[]                                    
                            for other_obj in gc.get_objects():
                                if isinstance(other_obj, HappyFolksAgent):
                                    idagent=other_obj.agent_id
                                    if (idagent not in visited):
                                        visited.append(other_obj.agent_id)    
                                        if (self.model.relations_matrix[self.agent_id,idagent]>=0.5) or (self.model.relations_matrix[partner.agent_id,idagent]>=0.5):
                                            self.add_to_latest_news("%s is sad after %s and %s suicide"%(other_obj.agent_name,partner.agent_name,self.agent_name))
                                            other_obj.happiness+=-2
                            
                            self.dead=1                       
                                
                            self.model.couples_matrix[self.agent_id,:]=np.nan
                            self.model.couples_matrix[:,self.agent_id]=np.nan
                            self.model.relations_matrix[self.agent_id,:]=np.nan
                            self.model.relations_matrix[:,self.agent_id]=np.nan 
                            
                            self.money=-1

                            partner.dead=1                       
                            
                            self.model.couples_matrix[partner.agent_id,:]=np.nan
                            self.model.couples_matrix[:,partner.agent_id]=np.nan
                            self.model.relations_matrix[partner.agent_id,:]=np.nan
                            self.model.relations_matrix[:,partner.agent_id]=np.nan 
                                            
                            partner.money=-1
                                        
            
            self.model.grid.move_to_empty(self)
            
        else:
            if self.model.verbose>=1:
                print("%s is now dead"%(self.agent_name))
                
            self.happiness=np.nan
            self.model.happiness_list[self.agent_id]=np.nan
            self.model.money_list[self.agent_id]=np.nan



class HappyFolks(Model):
    """
    Model class for the Schelling segregation model.
    """

    def __init__(self, height=10, width=10, n_agent=len(names),plotmat=False,plotmoney=False,
                 plotstep=1,matrix_init="pre_defined",money_init="pre_defined",verbose=1,
                 relation_like_happiness_var=1,relation_dislike_happiness_var=-1,external_disaster=False,
                 earthquake_proba=0.03,financialcrisis_proba=0.04,violence_list=[10,60,40,20,70,50],
                 business_risk=[0.2,0.1,0.4,0.2,0.5,0.1],gender_list=[1,1,1,0,0,0],happiness_list=[0,0,0,0,0,0],
                 sociability_list=[40,80,30,50,20,60]):
        
        """"""
                
        self.violence_list=violence_list
        self.business_risk=business_risk
        self.gender_list=gender_list
        self.happiness_list=happiness_list
        self.sociability_list=sociability_list
        self.external_disaster=external_disaster
        self.earthquake_proba=earthquake_proba
        self.financialcrisis_proba=financialcrisis_proba
        
        self.verbose=verbose

        self.height = height
        self.width = width
        self.n_agent = n_agent
        self.plotmat=plotmat
        self.plotmoney=plotmoney
        self.matrix_init=matrix_init
        self.money_init=money_init
        self.plotstep=plotstep

        self.relation_like_happiness_var=relation_like_happiness_var
        self.relation_dislike_happiness_var=relation_dislike_happiness_var
        
        self.schedule = RandomActivation(self)
        self.grid = SingleGrid(width, height, torus=True)

        self.model_happiness = 0
        self.relations_matrix=make_matrix_relations(names,mode=self.matrix_init)
        self.couples_matrix=np.zeros((n_agent,n_agent))
        self.happiness_list=[0]*self.n_agent
        
        self.money_list=make_money_list(names,mode=self.money_init)
        
        self.alive_agents=n_agent
        
        self.latest_news=""

        self.datacollector = DataCollector(
            {"model_happiness": "model_happiness"},  # Model-level happiness
            {"x": lambda a: a.pos[0], "y": lambda a: a.pos[1]}
        )
        
        grid_list=[]
        for cell in self.grid.coord_iter():
            x = cell[1]
            y = cell[2]
            grid_list.append([x,y])

        for i in range(self.n_agent):
            pos=random.choice(grid_list)
            grid_list.remove(pos)
            
            x=pos[0]
            y=pos[1]
            #print(x,y)
            
            agent_type = 0
            agent_id = i
            
            agent = HappyFolksAgent((x, y), self, agent_type,agent_id)
            self.grid.position_agent(agent, (x, y))
            self.schedule.add(agent)

        self.running = True
        self.datacollector.collect(self)
        self.stepnum=0
        
    def step(self):
        """
        Run one step of the model. If All agents are happy, halt the model.
        """
        #self.model_happiness = 0  # Reset counter of happy agents
        self.stepnum+=1

        self.latest_news=("Episode %.f"%(self.stepnum))
        
        self.alive_agents=0

        self.schedule.step()
        # collect data
        self.datacollector.collect(self)
        if self.verbose>=1:
            print("Model happiness :",self.model_happiness)
            
        if self.plotmat &(((self.stepnum)%(self.plotstep)==0)|(self.stepnum==1)):
            make_relations_matrix_plot(self.relations_matrix,title="Relations at step : %2.f"%(self.stepnum))
            make_relations_matrix_plot(self.couples_matrix,title="Couples at step : %2.f"%(self.stepnum))
        
        if (self.plotmoney)&(((self.stepnum)%(self.plotstep)==0)|(self.stepnum==1)):
            plt.figure(figsize=(10,10))
            #print(names[:self.n_agent],self.money_list[:self.n_agent])
            plt.bar(names[:self.n_agent],self.money_list[:self.n_agent])
            plt.title("Money at step : %2.f"%(self.stepnum))
            
        
        #relations_matrix_stock.append(self.relations_matrix.copy())
        #couples_matrix_stock.append(self.couples_matrix.copy())
        #money_list_stock.append(self.money_list[:])
        #happiness_list_stock.append(happiness_list[:])
        #n_alive_agents.append(self.alive_agents)
        #scenario.append(self.latest_news)
        
        #print(self.latest_news)
        
        if False:
            self.running = False

        return(self.relations_matrix,self.couples_matrix,self.money_list,self.happiness_list,self.alive_agents,self.latest_news)
       

#%% MultiRun the model for various violence lists



#%% Single run
