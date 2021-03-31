import os, sys, csv
import traci
import sumolib
import re
import configparser
import tensorflow as tf
import heapq
import numpy as np

from pathlib import Path
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
        self.model = tf.keras.models.load_model(os.path.join(current_dir, 'Model2'))
        if sys.platform == "linux" or sys.platform == "linux2":
            sim_dir = os.path.join(current_dir.parent, 'OSM/TESTING_SIMS/SUMO_FILES/')
            csv_dir = os.path.join(current_dir.parent, 'TEST_CSV/')
        elif sys.platform == "win32":
            sim_dir =  os.path.join(current_dir.parent, 'OSM\\TESTING_SIMS\\SUMO_FILES\\')
            csv_dir = os.path.join(current_dir.parent, 'TEST_CSV\\')

        self.simulationChoice = os.path.join(sim_dir, "map_1.sumocfg")
        self.net = sumolib.net.readNet(os.path.join(sim_dir, "map_1.net.xml"))
        self.netdict = {}

        now = datetime.now()
        format_date = str(now.strftime("%x")) + "_" + str(now.strftime("%X")) + "_" + str(now.strftime("%f"))
        self.output_file_name = csv_dir + format_date.replace("/", "-").replace(":", "-") + ".csv"

        self.run()

    def run(self):
        sumoBinary = os.path.join(os.environ['SUMO_HOME'], r'bin\sumo.exe')
        sumoCmd = [sumoBinary, "-c", self.simulationChoice]

        traci.start(sumoCmd)

        self.edges = self.getEdgeList()
        self.edges = [x for x in self.edges if ":" not in x]

        self.cost = {
        1 : 10,
        2 : 15,
        3 : 20,
        4 : 25,
        5 : 30,
        6 : 35,
        7 : 40,
        8 : 45
        }

        VEH_ID = "999999"
        vehicle_spawned = False
        self.step = 0
        start_edge = "37948285#4"
        end_edge = "-18651172#2"

        with open(self.output_file_name, 'a', newline='') as self.output:
            self.writer = csv.writer(self.output)

            while traci.simulation.getMinExpectedNumber() > 0:
                traci.simulationStep()
                self.step += 1
                print (self.step)
                if self.step == 300:
                    self.addVehicle(VEH_ID, start_edge)
                    self.initialRoute(VEH_ID, start_edge, end_edge)
                    vehicle_spawned = True

                # Code For Updating Routes

                if vehicle_spawned:
                    if (self.step % 20) == 0:
                        for edge in self.edges:
                            self.updateCosts(edge)
                            self.updateRoute(VEH_ID)

                    emissions = self.getEmissions(VEH_ID)
                    self.writer.writerow(emissions)


    def initialRoute(self, VEH_ID, start_edge, end_edge):
        route = traci.simulation.findRoute(start_edge, end_edge)
        traci.vehicle.setRoute(VEH_ID, route.edges)
        return

    def addVehicle(self, veh_id, start_edge):
        traci.route.add("InitialRoute", [start_edge])
        traci.vehicle.add(vehID=veh_id, routeID="InitialRoute")
        return

    def updateRoute(self, veh_id):
        traci.vehicle.rerouteTraveltime(veh_id, currentTravelTimes=False)
        return

    def getEdgeList(self):
        edges = traci.edge.getIDList()
        return edges

    def updateCosts(self, edge_id):
        get_edge = self.net.getEdge(edge_id)
        length = get_edge.getLength()

        speed = traci.edge.getLastStepMeanSpeed(edge_id)
        estimated_travel_time = traci.edge.getTraveltime(edge_id)
        traffic_level = traci.edge.getLastStepVehicleNumber(edge_id)

        input_data = [speed, estimated_travel_time, traffic_level, length]
        shaped_data = np.reshape(input_data, (4, 1)).T

        prediction = self.model.predict(shaped_data)
        pollution_class = np.argmax(prediction, axis=1)

        cost = self.cost[pollution_class[0]]
        updated_cost = estimated_travel_time + cost

        traci.edge.adaptTraveltime(edge_id, updated_cost)
        return

    def getNodes(self, edge_id):
        get_nodes = self.net.getEdge(edge_id)
        node_1_id = get_nodes.getFromNode().getID()
        node_2_id = get_nodes.getToNode().getID()
        return

    def getEmissions(self, veh_id):
        edge_id = traci.vehicle.getRoadID(veh_id)
        if edge_id != '':
            CO2_Emission = traci.edge.getCO2Emission(edge_id)
            CO_Emission = traci.edge.getCOEmission(edge_id)
            HC_Emission = traci.edge.getHCEmission(edge_id)
            NOx_Emission = traci.edge.getNOxEmission(edge_id)
            return [CO2_Emission, CO_Emission, HC_Emission, NOx_Emission]
        else:
            return [0, 0, 0, 0]

runSimulation()
