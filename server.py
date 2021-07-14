path=r'C:\Users\rapha\Desktop\Markov Chains & Agent Based Systems\Project\example'
import sys
sys.path.insert(1, path)

from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import CanvasGrid, ChartModule, TextElement
from mesa.visualization.UserParam import UserSettableParameter
from mesa.visualization.modules.BarChartVisualization import BarChartModule

from model import HappyFolks


class TotalHappiness(TextElement):
    """
    Display a text count of how many happy agents there are.
    """

    def __init__(self):
        pass

    def render(self, model):
        return "Overall happiness : " + str(model.model_happiness)


class LatestNews(TextElement):

    def __init__(self):
        pass

    def render(self, model):
        return "Latest news : " + str(model.latest_news)



def schelling_draw(agent):
    """
    Portrayal Method for canvas
    """
    if agent is None:
        return
    portrayal = {"Shape": "circle", "r": 0.5, "Filled": "true", "Layer": 0,"text_color":"Black"}

    if agent.dead==0:
        if agent.happiness<-10:
            portrayal["Color"] = ["firebrick", "maroon"]
            portrayal["stroke_color"] = "#000000"        
        elif agent.happiness >=-10 and agent.happiness < 0:
            portrayal["Color"] = ["lightcoral", "indianred"]
            portrayal["stroke_color"] = "#000000"
        elif agent.happiness >=0 and agent.happiness < 10:
            portrayal["Color"] = ["orange", "darkorange"]
            portrayal["stroke_color"] = "#000000"       
        elif agent.happiness >=10 and agent.happiness < 20:
            portrayal["Color"] = ["greenyellow", "lawngreen"]
            portrayal["stroke_color"] = "#000000"
        elif agent.happiness >=20:
            portrayal["Color"] = ["springgreen", "lime"]
            portrayal["stroke_color"] = "#000000"
    else:
            portrayal["Color"] = ["gray", "gray"]
            portrayal["stroke_color"] = "#000000"        
  
    
    portrayal["text"]=str(agent.happiness)+"|"+str(agent.agent_name)
    return portrayal


total_happiness = TotalHappiness()
latest_news = LatestNews()

canvas_element = CanvasGrid(schelling_draw, 10, 10, 500, 500)
happiness_chart = ChartModule([{"Label": "model_happiness", "Color": "black"}])


model_params = {
    "height": 10,
    "width": 10,
    "money_init":"random",
    "matrix_init":"random",
    "relation_like_happiness_var":UserSettableParameter('number', 'relation_like_happiness_var', value=1),
    "relation_dislike_happiness_var":UserSettableParameter('number', 'relation_dislike_happiness_var', value=-1)
    }

server = ModularServer(
    HappyFolks, [canvas_element,total_happiness,latest_news,happiness_chart], "HappyFolks", model_params
)
