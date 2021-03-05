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
        self.working_directory = os.getcwd() + r"\SUMO\\"

        self.config = (self.working_directory + r"OSM\TRAINING_SIMS\Config.ini")

        configParser = ConfigParser.RawConfigParser()
        configParser.read(self.config)

        self.simulationChoice = configParser['DEFAULT']['SimulationPath']
        self.net = sumolib.net.readNet(configParser['DEFAULT']['NetPath'])

    def run(self):
        sumoBinary = os.path.join(os.environ['SUMO_HOME'], r'bin\sumo.exe')
        sumoCmd = [sumoBinary, "-c", self.simulationChoice[0]]

        traci.start(sumoCmd)

        self.step = 0
        while traci.simulation.getMinExpectedNumber() > 0:
            break

    def addVehicle(self):
        return

    def intervalExec(self):
        return

    def staticDijkstra(self, nodes, start, finish):
        '''
        Static Dijkstra
        Attributes:
        listOfNodes: Lists Nodes to Search, FORMAT: ['A', 'B', 'C', 'D', ...]
        neighbourhood: List Distances From Node To Neighbours, FORMAT: {'A': {'B': 2, 'C': 4, 'D': 3}, 'B': {'A': 2, 'C': 3, 'D': 6}}
        startNode: Node where the vehicle is starting, FORMAT: 'A'
        '''

        unvisited = {node: float('inf') for node in listOfNodes} # Produce a dictionary with listOfNodes and set distance to infinity.
        print (unvisited)
        visited = {}
        current = startNode
        currentDistance = 0
        unvisited[current] = currentDistance

        while True:
            for neighbour, distance in neighbourhood[current].items():
                if neighbour not in unvisited: continue
                newDistance = currentDistance + distance
                if (unvisited[neighbour] == float('inf') or unvisited[neighbour] > newDistance):
                    unvisited[neighbour] = newDistance
            visited[current] = currentDistance
            del unvisited[current]
            if not unvisited: break
            candidates = [node for node in unvisited.items() if node[1]]
            current, currentDistance = sorted(candidates, key = lambda x: x[1])[0]
        return

    def dynamicDijkstra(self, nodes, start, finish):
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
