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
        self.emissionsmodel = tf.keras.models.load_model(os.path.join(current_dir, 'EmissionsModel.h5'))

        # Solves weird directory bugs during automation
        if sys.platform == "linux" or sys.platform == "linux2":
            sim_dir = os.path.join(current_dir.parent, 'OSM/TESTING_SIMS/SUMO_FILES/')
            csv_dir = os.path.join(current_dir.parent, 'TEST_CSV/')
        elif sys.platform == "win32":
            sim_dir =  os.path.join(current_dir.parent, 'OSM\\TESTING_SIMS\\SUMO_FILES\\')
            csv_dir = os.path.join(current_dir.parent, 'TEST_CSV\\')

        # Run All Tests Without Supervision
        base = 1
        # Dijkstra, DynamicDijkstra, GlobalCost
        self.options = [True, False, True]
        for i in range(1):
            mapfile = "map_" + str((base+i)) # Identifies which map to run tests on automatically
            file1 = str(mapfile) + ".sumocfg"
            file2 = str(mapfile) + ".net.xml"

            # Defines simulation and network file locations
            self.simulationChoice = os.path.join(sim_dir, file1)
            self.net = sumolib.net.readNet(os.path.join(sim_dir, file2))

            # Contains total number of neighbours per edge, KEY: EdgeID
            self.neighbours = {}


            # Pre-processed edge ids
            self.listed_edges = self.net.getEdges()
            # Post-processed edge ids
            self.edges = []

            # Updating the same list with the new EdgeID's breaks calcNeighbours() so a second list is used
            for edge in self.listed_edges: # Alternative was to use range and indexing, but creates bug for calculate neighbours
                edge_index = self.listed_edges.index(edge)
                edge = edge.getID() # Get's traci ID for each edge
                self.neighbours[edge] = self.calcNeighbours(self.net, edge)
                self.edges.append(edge)

            # Eliminates false edges created by SUMO's edge detection system
            self.edges = [x for x in self.edges if ":" not in x]

            # Holds the beginning and end edges to create routes for vehicle tracking
            self.start_edges = []
            self.end_edges = []

            # Contains cost for each pollution class
            self.cost = {
            1 : 1,
            2 : 1,
            3 : 1,
            4 : 1,
            5 : 1,
            6 : 1,
            7 : 1,
            8 : 1,
            9 : 1,
            10 : 100,
            11 : 100,
            12 : 100
            }

            for i in range(10):
                self.edgeGenerator()

            # Run simulation for each posisble route
            for self.edge_index in range(len(self.start_edges)):
                self.start_edge = self.start_edges[self.edge_index]
                self.end_edge = self.end_edges[self.edge_index]

                # Run simulation multiple times for each possible routing method
                for i in range(4):
                    self.dijkstra = self.options[0]
                    self.dynamic_dijkstra = self.options[1]
                    self.global_cost = self.options[2]

                    if (self.dijkstra and self.global_cost):
                        self.output_file_name = csv_dir + mapfile + "_SDGC_" + str(i) + str(self.edge_index) + ".csv"
                        self.options[0] = False
                        self.options[1] = True
                    elif (self.dynamic_dijkstra and self.global_cost):
                        self.output_file_name = csv_dir + mapfile + "_DDGC_" + str(i) + str(self.edge_index) + ".csv"
                        self.options[0] = True
                        self.options[1] = False
                        self.options[2] = False
                    elif self.dijkstra:
                        self.output_file_name = csv_dir + mapfile + "_D_" + str(i) + str(self.edge_index) + ".csv"
                        self.options[0] = False
                        self.options[1] = True
                    elif self.dynamic_dijkstra:
                        self.output_file_name = csv_dir + mapfile + "_DD_" + str(i) + str(self.edge_index) + ".csv"
                        self.options[0] = True
                        self.options[1] = False
                        self.options[2] = True

                    final_edges = self.run()
                    with open((csv_dir+mapfile+'edgesused.csv'), 'a', newline='') as edgeOutput:
                        edgewriter = csv.writer(edgeOutput)
                        edgewriter.writerow(final_edges)
                    edgeOutput.close()

                # Used to catch errors from edge generator as valid routes can only be determined when the simulation attempts to add vehicle
                if (self.tripped1 or self.tripped2) == "Error":
                    self.edgeGenerator()
                    self.tripped1 = ""
                    self.tripped2 = ""


    def run(self):
        if sys.platform == "linux" or sys.platform == "linux2":
            sumoBinary = os.path.join(os.environ['SUMO_HOME'], r'bin/sumo')
        elif sys.platform == "win32":
            sumoBinary = os.path.join(os.environ['SUMO_HOME'], r'bin\sumo.exe')

        sumoCmd = [sumoBinary, "-c", self.simulationChoice]

        traci.start(sumoCmd)

        # Keeps log of edges and predicted pollution classes
        # Edge : Pollution Class, Counter
        self.rank_log = {edge : [1, 0] for edge in self.edges}

        self.VEH_ID = "999999" # Spawned vehicle ID
        vehicle_spawned = False # Variable to determine if vehicle spawned
        self.step = traci.simulation.getTime() # Current step
        self.previousDistance = 99999
        spawnstep = self.step + 500 # What step to spawn the vehicle
        self.dynamic_interval = 25 # How often a route should be updated, changes during simulation
        self.routeInitialised = False
        route_found = False # Used to determine if route is real

        # Catch false routes (two separate methods required due to bugs)
        while not route_found:
            try:
                self.verifyRoute(self.start_edge, self.end_edge)
                route_found = True
            except:
                self.edgeGenerator(override=True)
                self.start_edge = self.start_edges[self.edge_index]
                self.end_edge = self.end_edges[self.edge_index]

        # Creates new file log for each routing method
        with open(self.output_file_name, 'w', newline='') as self.output:
            self.writer = csv.writer(self.output)

            while traci.simulation.getMinExpectedNumber() > 0:
                traci.simulationStep() # Increment simulation step
                self.step += 1 # Track simulation step

                # Catch errors with false routes
                if self.step == spawnstep:
                    self.tripped1 = self.addVehicle(self.VEH_ID, self.start_edge)
                    self.tripped2 = self.self.routeInitialised(self.VEH_ID, self.start_edge, self.end_edge)
                    if (self.tripped1 or self.tripped2) == "Error":
                        traci.close()
                        break
                    else:
                        vehicle_spawned = True

                # Update routing method based on routing choice
                if vehicle_spawned:
                    if (self.dijkstra and self.global_cost):
                        self.dijkstra_global()
                    elif (self.dynamic_dijkstra and self.global_cost):
                        self.dyn_dijkstra_global()
                    elif self.dijkstra:
                        self.dijkstra_std()
                    elif self.dynamic_dijkstra:
                        self.dyn_dijkstra_std()

                    try:
                        current_edge = traci.vehicle.getRoadID(self.VEH_ID)
                    except traci.exceptions.TraCIException:
                        traci.close()
                        break

                    if current_edge != self.end_edge:
                        emissions = self.getEmissions(self.VEH_ID)
                        self.writer.writerow(emissions)
                    else:
                        traci.close()
                        break

            self.output.close()

            return [self.start_edge, self.end_edge]

    def dijkstra_global(self):
        print ("\nDijkstra & Global Cost")
        if self.step == spawnstep:
            for edge in self.edges:
                self.updateCosts(edge)
            self.updateRoute(self.VEH_ID, False)
        return

    def dyn_dijkstra_global(self):
        print ("\nDynamic Dijkstra & Global Cost")
        try:
            if (traci.vehicle.getRoadID(self.VEH_ID) != "") and (self.routeInitialised == False):
                for edge in self.edges:
                    self.updateCosts(edge)
                self.updateRoute(self.VEH_ID, False)
                self.routeInitialised = True
                distance = self.getDistance(self.VEH_ID, self.end_edge)
                self.dynamic_interval = self.intervalUpdater(distance)
            elif (traci.vehicle.getRoadID(self.VEH_ID) != "") and (self.routeInitialised == True):
                if ((self.step % 25) == 0): #self.dynamic_interval instead of 25
                    print (self.rank_log)
                    for edge in self.edges:
                        self.updateCosts(edge)
                    self.updateRoute(self.VEH_ID, False)
                    distance = self.getDistance(self.VEH_ID, self.end_edge)
                    self.dynamic_interval = self.intervalUpdater(distance)
                    print ("\nEdge ->", traci.vehicle.getRoadID(self.VEH_ID))
                    print ("\nDistance -> ", distance)
        except:
            pass
        return

    def dijkstra_std(self):
        print ("\nDijkstra")
        if self.step == spawnstep:
            self.updateRoute(self.VEH_ID, True)
        return

    def dyn_dijkstra_std(self):
        print ("\nDynamic Dijkstra")
        if ((self.step % 25) == 0) or (self.step == spawnstep):
            self.updateRoute(self.VEH_ID, True)
        return


    def edgeGenerator(self, override=False):
        'Generates start and end edges randomly'
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
        'Verifies if start and end edges are connected and drivable'
        route = traci.simulation.findRoute(start_edge, end_edge)
        if route.edges == ():
            raise RoutingError()
        return

    def initialRoute(self, veh_id, start_edge, end_edge):
        'Finds an initial route, subject to change depending on routing method'
        route = traci.simulation.findRoute(start_edge, end_edge)
        traci.vehicle.setRoute(veh_id, route.edges)
        return

    def addVehicle(self, veh_id, start_edge):
        'Adds vehicle to simulation'
        traci.route.add("InitialRoute", [start_edge])
        try:
            traci.vehicle.add(vehID=veh_id, routeID="InitialRoute")
        except traci.exceptions.TraCIException:
            return "Error"
        return

    def getEdgeList(self):
        'Creates list of edges'
        edges = traci.edge.getIDList()
        return edges


    def updateCosts(self, edge_id):
        'Complex cost function using neural networks to determine route costs'
        get_edge = self.net.getEdge(edge_id)
        length = get_edge.getLength()
        speed = traci.edge.getLastStepMeanSpeed(edge_id)
        estimated_travel_time = traci.edge.getTraveltime(edge_id)
        total_neighbours = self.neighbours.get(edge_id)

        input_data = [speed, estimated_travel_time, total_neighbours, length]
        shaped_data = np.reshape(input_data, (4, 1)).T

        current_rank = self.rank_log.get(edge_id)
        if current_rank[1] == 3:
            current_rank[0] = 1
            current_rank[1] = 0

        model_pred = self.emissionsmodel.predict(shaped_data)
        prediction = np.argmax(model_pred, axis=1)
        pollution_class = prediction[0]

        # Short Term Memory Implementation
        if current_rank[1] < 2:
            if pollution_class <= current_rank[0]:
                pollution_class = current_rank[0]

        cost = self.cost[pollution_class]
        updated_cost = (cost*estimated_travel_time)

        traci.edge.adaptTraveltime(edge_id, updated_cost)
        current_rank = [pollution_class, (current_rank[1]+1)]
        self.rank_log[edge_id] = current_rank
        return

    def updateRoute(self, veh_id, updateTimes):
        'Update travel times'
        traci.vehicle.rerouteTraveltime(veh_id, currentTravelTimes=updateTimes)
        return

    def getNodes(self, edge_id):
        'Gets nodes on either side of edge'
        get_nodes = self.net.getEdge(edge_id)
        node_1_id = get_nodes.getFromNode().getID()
        node_2_id = get_nodes.getToNode().getID()
        return

    def getEmissions(self, veh_id):
        'Tracks live emissions on each edge the tracked vehicle is on'
        edge_id = traci.vehicle.getRoadID(veh_id)
        if edge_id != '':
            CO2_Emission = traci.edge.getCO2Emission(edge_id)
            CO_Emission = traci.edge.getCOEmission(edge_id)
            HC_Emission = traci.edge.getHCEmission(edge_id)
            NOx_Emission = traci.edge.getNOxEmission(edge_id)
            PMx_Emission = traci.edge.getPMxEmission(edge_id)
            return [CO2_Emission, CO_Emission, HC_Emission, NOx_Emission, PMx_Emission]
        else:
            return [0, 0, 0, 0, 0]

    def getDistance(self, veh_id, end_edge):
        'Gets vehicles distance from end edge'
        current_edge_id = traci.vehicle.getRoadID(veh_id)
        current_edge_id = current_edge_id.replace(":", "")
        current_edge_id = current_edge_id.split("_")

        # Sometimes SUMO returns a false edge, hence a check is used to catch this
        try:
            current_edge = self.net.getEdge(current_edge_id[0])
        except KeyError:
            return self.previousDistance

        node_2_id = current_edge.getToNode().getID()
        x, y = self.net.getNode(node_2_id).getCoord()

        final_edge = self.net.getEdge(end_edge)
        node_1_id = final_edge.getToNode().getID()
        x1, y1 = self.net.getNode(node_1_id).getCoord()

        distance = traci.simulation.getDistance2D(x, y, x1, y1, isGeo=False, isDriving=True)
        self.previousDistance = distance
        return distance

    def intervalUpdater(self, distance):
        'Dynamic interval updator based on distance to final destination'
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

    def calcNeighbours(self, net, edge):
        'Calculates neighbouring edges to specified edge'
        get_edge = net.getEdge(edge)
        node_1_id = get_edge.getFromNode().getID()
        node_2_id = get_edge.getToNode().getID()
        node_1_coord = net.getNode(node_1_id).getCoord()
        node_2_coord = net.getNode(node_2_id).getCoord()
        node_1_ne = net.getNeighboringEdges(node_1_coord[0], node_1_coord[1], r=0.0001)
        node_2_ne = net.getNeighboringEdges(node_2_coord[0], node_2_coord[1], r=0.0001)
        node_1_ne = len(node_1_ne)
        node_2_ne = len(node_2_ne)
        return (int(node_1_ne) + int(node_2_ne))

runSimulation()
