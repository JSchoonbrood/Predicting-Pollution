import os, sys, csv
import traci
import sumolib
import re
import configparser
import heapq

from datetime import datetime
from threading import Thread

if 'SUMO_HOME' in os.environ: # Assumes Environment Path Declared
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

class runSimulation():
    def __init__(self):
        current_dir = Path(os.path.dirname(__file__))
        if sys.platform == "linux" or sys.platform == "linux2":
            sim_dir = os.path.join(current_dir.parent, 'OSM/TRAINING_SIMS/SUMO_FILES/')
            csv_dir = os.path.join(current_dir.parent, 'CSV/')
        elif sys.platform == "win32":
            sim_dir =  os.path.join(current_dir.parent, 'OSM\\TRAINING_SIMS\\SUMO_FILES\\')
            csv_dir = os.path.join(current_dir.parent, 'CSV\\')

        self.simulationChoice = os.path.join(sim_dir, "TestSimulation.sumocfg")
        self.net = sumolib.net.readNet(os.path.join(sim_dir, "TestSimulation.net.xml"))
        self.netdict = {}

        now = datetime.now()
        format_date = str(now.strftime("%x")) + "_" + str(now.strftime("%X")) + "_" + str(now.strftime("%f"))
        self.output_file_name = csv_dir + format_date.replace("/", "-").replace(":", "-") + ".csv"

        self.run()

    def run(self):
        sumoBinary = os.path.join(os.environ['SUMO_HOME'], r'bin\sumo.exe')
        sumoCmd = [sumoBinary, "-c", self.simulationChoice]

        traci.start(sumoCmd)

        self.step = 0
        while traci.simulation.getMinExpectedNumber() > 0:
            break

    def initialRoute(self):
        return

    def updateRoute(self):
        return

    def addVehicle(self):
        return

    def intervalExec(self):
        return

    def aStar(self, nodes, start, finish):
        return

    def aStarModified(self, nodes, start, finish):
        return

    def giveCosts(self):
        '''UPDATE EDGE COSTS'''
        return

    def getEdgeList(self):
        edges = traci.edge.getIDList()
        return edges

    def getNodes(self, edge_id):
        get_nodes = self.net.getEdge(edge_id)
        node_1_id = get_nodes.getFromNode().getID()
        node_2_id = get_nodes.getToNode().getID()
        return
