import os
import sys
import csv
import traci
import sumolib
from pathlib import Path
from datetime import datetime

if 'SUMO_HOME' in os.environ:  # Assumes Environment Path Declared
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")


class runSimulation():
    def __init__(self):
        # Assign Directories
        current_dir = Path(os.path.dirname(__file__))
        if sys.platform == "linux" or sys.platform == "linux2":
            sim_dir = os.path.join(current_dir.parent,
                                   'OSM/TRAINING_SIMS/SUMO_FILES/')
            csv_dir = os.path.join(current_dir.parent, 'CSV/')
        elif sys.platform == "win32":
            sim_dir = os.path.join(current_dir.parent,
                                   'OSM\\TRAINING_SIMS\\SUMO_FILES\\')
            csv_dir = os.path.join(current_dir.parent, 'CSV\\')

        # Automated Process To Process Multiple Simulations
        base = 1
        for i in range(9):
            mapfile = "map_" + str((base+i))
            file1 = str(mapfile) + ".sumocfg"
            file2 = str(mapfile) + ".net.xml"
            self.simulationChoice = os.path.join(sim_dir, file1)
            self.net = sumolib.net.readNet(os.path.join(sim_dir, file2))
            self.netdict = {}

            # Create Unique Datafile Name
            now = datetime.now()
            format_date = (str(now.strftime("%x")) + "_"
                           + str(now.strftime("%X")) + "_"
                           + str(now.strftime("%f")))

            self.output_file_name = (csv_dir + format_date.replace("/", "-").
                                     replace(":", "-") + ".csv")

            # Create Datafile
            with open(self.output_file_name, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Edge", "CO2_Emissions", "CO_Emissions",
                                 "HC_Emissions", "NOx_Emissions", "PMx_Emissions",
                                 "Mean_Speed", "Estimated_Travel_Time", "Traffic_Level",
                                 "Total_Neighbours", "Length", "Step_Count"])
                file.close()

            self.run()

    def run(self):
        sumoBinary = os.path.join(os.environ['SUMO_HOME'], r'bin\sumo.exe')
        sumoCmd = [sumoBinary, "-c", self.simulationChoice]

        # Starts simulation
        traci.start(sumoCmd)

        # Creates a list of edges
        self.edges = self.getEdgeList()

        self.edges = [x for x in self.edges if ":" not in x]
        for edge in self.edges:
            # SUMO returns false edges from getIDList(), this will filter them
            self.netdict[edge] = self.calculateNeighbours(edge)
            self.netdict[edge] = self.calculateEdgeLengths(edge)

        with open(self.output_file_name, 'a', newline='') as self.output:
            self.writer = csv.writer(self.output)

            self.step = 0
            while traci.simulation.getMinExpectedNumber() > 0:
                # if self.step == 0:
                traci.simulationStep()
                self.step += 1
                '''elif self.step > 0:
                    traci.simulationStep()
                    self.step += 1'''
                if (self.step % 20) == 0:
                    for row in self.intervalData():
                        self.writer.writerow(row)

        traci.close()
        self.output.close()

    def getEdgeList(self):
        edges = traci.edge.getIDList()
        return edges

    def getCurrentEdge(self, ID):
        '''We will have this data in real life.'''
        currentEdge = traci.vehicle.getRoadID(ID)
        return currentEdge

    def getEmissions(self, edge_id):
        '''Note: May output 0, as if no vehicles are along an edge in the
        previous time step it will output 0.'''
        CO2_Emission = traci.edge.getCO2Emission(edge_id)
        CO_Emission = traci.edge.getCOEmission(edge_id)
        HC_Emission = traci.edge.getHCEmission(edge_id)
        NOx_Emission = traci.edge.getNOxEmission(edge_id)
        PMx_Emission = traci.edge.getPMxEmission(edge_id)
        # print ([CO2_Emission, CO_Emission, HC_Emission, NOx_Emission])
        return [CO2_Emission, CO_Emission, HC_Emission, NOx_Emission, PMx_Emission]

    def getEdgeTravelTime(self, edge_id):
        '''Edge travel time would be a prediction we can make in real life.'''
        length = traci.edge.getTraveltime(edge_id)
        return length

    def getLaneDistance(self, edge_id):
        '''We will have this data in real life.'''
        distance = traci.lane.getLength(edge_id)
        return distance

    def getEdgeWaitingTime(self, edge_id):
        time = traci.edge.getWaitingTime(edge_id)
        return time

    def getEdgeDensity(self, edge_id):
        '''This is a value we will be able to predict in real life based on
        mean speed, gps position along the route, acceleration / deceleration
        data, average gap between vehicles, and possible data from other
        vehicles along the edge using the same app.'''
        vehicle_quantity = traci.edge.getLastStepVehicleNumber(edge_id)
        return vehicle_quantity

    def getEdgeSpeed(self, edge_id):
        '''We're using mean speed of the vehicle rather than the edge because
        in a real life situation we would not have edge mean speed.'''
        speed = traci.edge.getLastStepMeanSpeed(edge_id)
        # mean_speed = traci.edge.getLastStepMeanSpeed(edge_id)
        return speed

    def calculateNeighbours(self, edge_id):
        get_edge = self.net.getEdge(edge_id)
        node_1_id = get_edge.getFromNode().getID()
        node_2_id = get_edge.getToNode().getID()
        node_1_coord = self.net.getNode(node_1_id).getCoord()
        node_2_coord = self.net.getNode(node_2_id).getCoord()
        node_1_ne = self.net.getNeighboringEdges(node_1_coord[0],
                                                 node_1_coord[1], r=0.0001)
        node_2_ne = self.net.getNeighboringEdges(node_2_coord[0],
                                                 node_2_coord[1], r=0.0001)
        node_1_ne = len(node_1_ne)
        node_2_ne = len(node_2_ne)
        return (int(node_1_ne) + int(node_2_ne))

    def calculateEdgeLengths(self, edge_id):
        get_edge = self.net.getEdge(edge_id)
        get_length = get_edge.getLength()
        return [self.netdict[edge_id], get_length]

    def intervalData(self):
        rows = []
        for edge in self.edges:
            current_edge = str(edge)
            emissions = self.getEmissions(current_edge)
            mean_speed = self.getEdgeSpeed(current_edge)
            estimated_travel_time = self.getEdgeTravelTime(current_edge)
            traffic_level = self.getEdgeDensity(current_edge)
            values = self.netdict[current_edge]
            neighbours = values[0]
            length = values[1]

            current_row = [current_edge, emissions[0], emissions[1],
                           emissions[2], emissions[3], emissions[4], mean_speed,
                           estimated_travel_time, traffic_level, neighbours,
                           length, self.step]
            rows.append(current_row)
        return rows


runSimulation()
