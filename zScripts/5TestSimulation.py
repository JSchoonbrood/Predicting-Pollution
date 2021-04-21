import os, sys, csv
import traci
import sumolib
import re
import configparser
import tensorflow as tf
import heapq
import numpy as np

from random import randint
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
        self.model1 = tf.keras.models.load_model(os.path.join(current_dir, 'Model1'))
        self.model2 = tf.keras.models.load_model(os.path.join(current_dir, 'Model2-5'))
        if sys.platform == "linux" or sys.platform == "linux2":
            sim_dir = os.path.join(current_dir.parent, 'OSM/TESTING_SIMS/SUMO_FILES/')
            csv_dir = os.path.join(current_dir.parent, 'TEST_CSV/')
        elif sys.platform == "win32":
            sim_dir =  os.path.join(current_dir.parent, 'OSM\\TESTING_SIMS\\SUMO_FILES\\')
            csv_dir = os.path.join(current_dir.parent, 'TEST_CSV\\')

        # Run All Tests Without Supervision
        base = 1
        # Dijkstra, DynamicDijkstra, GlobalCost, LocalCost
        self.options = [True, False, True, False]
        for i in range(1):
            mapfile = "map_" + str((base+i))
            file1 = str(mapfile) + ".sumocfg"
            file2 = str(mapfile) + ".net.xml"

            self.simulationChoice = os.path.join(sim_dir, file1)
            self.net = sumolib.net.readNet(os.path.join(sim_dir, file2))
            self.netdict = {}

            self.edges = self.net.getEdges()
            for i in range(len(self.edges)):
                self.edges[i] = self.edges[i].getID()
            self.edges = [x for x in self.edges if ":" not in x]

            self.start_edges = []
            self.end_edges = []

            for i in range(10):
                self.edgeGenerator()


            self.cost = {
            1 : 0,
            2 : 10,
            3 : 20,
            4 : 25,
            5 : 30,
            6 : 35,
            7 : 50,
            8 : 60
            }

            for self.edge_index in range(len(self.start_edges)):
                self.start_edge = self.start_edges[self.edge_index]
                self.end_edge = self.end_edges[self.edge_index]

                for i in range(6):
                    self.dijkstra = self.options[0]
                    self.dynamic_dijkstra = self.options[1]
                    self.global_cost = self.options[2]
                    self.local_cost = self.options[3]

                    if (self.dijkstra and self.global_cost):
                        self.output_file_name = csv_dir + mapfile + "_SDGC_" + str(i) + str(self.edge_index) + ".csv"
                        self.options[0] = False
                        self.options[1] = True
                    elif (self.dynamic_dijkstra and self.global_cost):
                        self.output_file_name = csv_dir + mapfile + "_DDGC_" + str(i) + str(self.edge_index) + ".csv"
                        self.options[2] = False
                        self.options[3] = True
                    elif (self.dynamic_dijkstra and self.local_cost):
                        self.output_file_name = csv_dir + mapfile + "_DDLC_" + str(i) + str(self.edge_index) + ".csv"
                        self.options[0] = True
                        self.options[1] = False
                    elif (self.dijkstra and self.local_cost):
                        self.output_file_name = csv_dir + mapfile + "_SDLC_" + str(i) + str(self.edge_index) + ".csv"
                        self.options[0] = True
                        self.options[1] = False
                        self.options[2] = False
                        self.options[3] = False
                    elif self.dijkstra:
                        self.output_file_name = csv_dir + mapfile + "_D_" + str(i) + str(self.edge_index) + ".csv"
                        self.options[0] = False
                        self.options[1] = True
                    elif self.dynamic_dijkstra:
                        self.output_file_name = csv_dir + mapfile + "_DD_" + str(i) + str(self.edge_index) + ".csv"
                        self.options[0] = True
                        self.options[1] = False
                        self.options[2] = True

                    self.run()

                if (self.tripped1 or self.tripped2) == "Error":
                    self.edgeGenerator()
                    self.tripped1 = ""
                    self.tripped2 = ""


    def run(self):
        sumoBinary = os.path.join(os.environ['SUMO_HOME'], r'bin\sumo.exe')
        sumoCmd = [sumoBinary, "-c", self.simulationChoice]

        traci.start(sumoCmd)

        VEH_ID = "999999"
        vehicle_spawned = False
        self.step = traci.simulation.getTime()
        spawnstep = self.step + 500
        dynamic_interval = 25
        initialRoute = False

        route_found = False
        while not route_found:
            try:
                self.verifyRoute(self.start_edge, self.end_edge)
                route_found = True
            except:
                self.edgeGenerator(override=True)
                self.start_edge = self.start_edges[self.edge_index]
                self.end_edge = self.end_edges[self.edge_index]

        with open(self.output_file_name, 'w', newline='') as self.output:
            self.writer = csv.writer(self.output)

            while traci.simulation.getMinExpectedNumber() > 0:
                traci.simulationStep()
                self.step += 1

                if self.step == spawnstep:
                    self.tripped1 = self.addVehicle(VEH_ID, self.start_edge)
                    self.tripped2 = self.initialRoute(VEH_ID, self.start_edge, self.end_edge)
                    if (self.tripped1 or self.tripped2) == "Error":
                        traci.close()
                        break
                    else:
                        vehicle_spawned = True

                if vehicle_spawned:


                    if (self.dijkstra and self.global_cost):
                        print ("Dijkstra & Global Cost")
                        if self.step == spawnstep:
                            for edge in self.edges:
                                self.updateCosts(edge)
                            self.updateRoute(VEH_ID, False)


                    elif (self.dynamic_dijkstra and self.global_cost):
                        print ("Dynamic Dijkstra & Global Cost")
                        if (traci.vehicle.getRoadID(VEH_ID) != "") and (initialRoute == False):
                            for edge in self.edges:
                                self.updateCosts(edge)
                            self.updateRoute(VEH_ID, False)
                            initialRoute = True
                            distance = self.getDistance(VEH_ID, self.end_edge)
                            self.dynamic_interval = self.intervalUpdater(distance)
                        elif (traci.vehicle.getRoadID(VEH_ID) != "") and (initialRoute == True):
                            if ((self.step % self.dynamic_interval) == 0):
                                    for edge in self.edges:
                                        self.updateCosts(edge)
                                    self.updateRoute(VEH_ID, False)
                                    distance = self.getDistance(VEH_ID, self.end_edge)
                                    self.dynamic_interval = self.intervalUpdater(distance)
                                    print ("\nEdge ->", traci.vehicle.getRoadID(VEH_ID))
                                    print ("\nDistance -> ", distance)


                    elif (self.dynamic_dijkstra and self.local_cost):
                        print ("Dynamic Dijkstra & Local Cost")
                        if ((self.step % 25) == 0) or (self.step == spawnstep):
                            for edge in self.edges:
                                self.updatelocalCosts(edge)
                            self.updateRoute(VEH_ID, False)
                    elif (self.dijkstra and self.local_cost):
                        print ("Dijkstra & Local Cost")
                        if self.step == spawnstep:
                            for edge in self.edges:
                                self.updatelocalCosts(edge)
                            self.updateRoute(VEH_ID, False)
                    elif self.dijkstra:
                        print ("Dijkstra")
                        if self.step == spawnstep:
                            self.updateRoute(VEH_ID, True)
                    elif self.dynamic_dijkstra:
                        print ("Dynamic Dijkstra")
                        if ((self.step % 25) == 0) or (self.step == spawnstep):
                            self.updateRoute(VEH_ID, True)
                    try:
                        current_edge = traci.vehicle.getRoadID(VEH_ID)
                    except traci.exceptions.TraCIException:
                        traci.close()
                        break

                    if current_edge != self.end_edge:
                        emissions = self.getEmissions(VEH_ID)
                        self.writer.writerow(emissions)
                    else:
                        traci.close()
                        break

            self.output.close()

    def edgeGenerator(self, override=False):
        edge_1 = randint(0, len(self.edges))
        edge_2 = randint(0, len(self.edges))

        if override == True:
            self.start_edges[self.edge_index] = self.edges[edge_1]
            self.end_edges[self.edge_index] = self.edges[edge_2]
        else:
            self.start_edges.append(self.edges[edge_1])
            self.end_edges.append(self.edges[edge_2])
        return

    def verifyRoute(self, start_edge, end_edge):
        route = traci.simulation.findRoute(start_edge, end_edge)
        if route.edges == ():
            raise RoutingError()
        return

    def initialRoute(self, VEH_ID, start_edge, end_edge):
        route = traci.simulation.findRoute(start_edge, end_edge)
        traci.vehicle.setRoute(VEH_ID, route.edges)
        return

    def addVehicle(self, veh_id, start_edge):
        traci.route.add("InitialRoute", [start_edge])
        try:
            traci.vehicle.add(vehID=veh_id, routeID="InitialRoute")
        except traci.exceptions.TraCIException:
            return "Error"
        return

    def getEdgeList(self):
        edges = traci.edge.getIDList()
        return edges

    def updatelocalCosts(self, edge_id):
        get_edge = self.net.getEdge(edge_id)
        length = get_edge.getLength()

        estimated_travel_time = traci.edge.getTraveltime(edge_id)

        traffic_level = traci.edge.getLastStepVehicleNumber(edge_id)

        cost = 0.1*(traffic_level*estimated_travel_time) + 1*(estimated_travel_time)

        traci.edge.adaptTraveltime(edge_id, cost)
        return

    def updateCosts(self, edge_id):
        get_edge = self.net.getEdge(edge_id)
        length = get_edge.getLength()

        speed = traci.edge.getLastStepMeanSpeed(edge_id)
        estimated_travel_time = traci.edge.getTraveltime(edge_id)
        traffic_level = traci.edge.getLastStepVehicleNumber(edge_id)

        input_data = [speed, estimated_travel_time, traffic_level, length]
        shaped_data = np.reshape(input_data, (4, 1)).T

        prediction1 = self.model1.predict(shaped_data)
        pollution_class = np.argmax(prediction1, axis=1)
        if pollution_class[0] != 2:
            prediction = self.model2.predict(shaped_data)
            pollution_class = np.argmax(prediction, axis=1)
        else:
            pollution_class = [8]

        cost = self.cost[pollution_class[0]]
        updated_cost = 1*(cost*estimated_travel_time) + 1*(estimated_travel_time)

        traci.edge.adaptTraveltime(edge_id, updated_cost)
        return

    def updateRoute(self, veh_id, updateTimes):
        traci.vehicle.rerouteTraveltime(veh_id, currentTravelTimes=updateTimes)
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

    def getDistance(self, veh_id, end_edge):
        current_edge_id = traci.vehicle.getRoadID(veh_id)
        current_edge_id = current_edge_id.replace(":", "")
        current_edge_id = current_edge_id.split("_")
        current_edge = self.net.getEdge(current_edge_id[0])
        node_2_id = current_edge.getToNode().getID()
        x, y = self.net.getNode(node_2_id).getCoord()

        final_edge = self.net.getEdge(end_edge)
        node_1_id = final_edge.getToNode().getID()
        x1, y1 = self.net.getNode(node_1_id).getCoord()


        distance = traci.simulation.getDistance2D(x, y, x1, y1, isGeo=False, isDriving=True)

        return distance

    def intervalUpdater(self, distance):
        if distance > 5000:
            interval = 80
        elif distance > 2500:
            interval = 60
        elif distance > 1000:
            interval = 35
        elif distance < 1000:
            interval = 25

        print ("\nInterval -> ", interval)
        return interval

runSimulation()
