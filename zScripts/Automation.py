import os
import sys

from pathlib import Path
from sys import platform

working_directory = r"C:\\Users\\Jake\\Desktop\\Projects\\SUMO\\OSM\\TRAINING_SIMS\\SUMO_FILES\\"
sys.path.insert(1, working_directory)
import randomTrips
#os.system('cmd /k "Your Command Prompt Command"')


def osmToXML():
    current_dir = Path(os.path.dirname(__file__))
    if sys.platform == "linux" or sys.platform == "linux2":
        data_dir = os.path.join(current_dir.parent, 'OSM/TRAINING_SIMS/')
    elif sys.platform == "win32":
        data_dir =  os.path.join(current_dir.parent, 'OSM\\TRAINING_SIMS\\')

    osm_files = os.listdir(data_dir)
    for fname in osm_files:
        path = os.path.join(data_dir, fname)
        if os.path.isdir(path):
            continue

        path = os.path.join(data_dir, fname)
        new_path = os.path.join((data_dir + "SUMO_FILES\\"), fname)
        command = ("netconvert --osm-files " + path + " -o " +
                    new_path.replace(".osm", ".net.xml"))
        os.system('cmd /c ' + str(command))

        with open(new_path.replace(".osm", ".sumocfg"), 'wt') as file:
            file.flush()
            file.write("<configuration>\n\n")
            file.write("<input>\n")
            file.write(str('<net-file value="' + new_path.replace(".osm", ".net.xml" + '"/>\n')))
            file.write(str('<route-files value="' + new_path.replace(".osm", ".rou.xml" + '"/>\n')))
            file.write("</input>\n")
            file.write("<time>\n")
            file.write('<begin value = "0"/>\n')
            file.write('<end value = "3600"/>\n')
            file.write('</time>\n\n')
            file.write('</configuration>\n')
            file.flush()
        file.close()

    return osm_files



def ranTripGen(osm_files):
    number = "14859"
    working_directory = r"C:\\Users\\Jake\\Desktop\\Projects\\SUMO\\OSM\\TRAINING_SIMS\\SUMO_FILES\\"
    #from working_directory import randomTrips
    #import (str(working_directory + "randomTrips.py"))
    for file in osm_files:
        file.replace(".osm", ".net.xml")

        randomTrips.get_options((("-n " + file + " -r " + file.replace(".net.xml", ".rou.xml") + " -e " + number + " l ")))

        #randomTrips()
        #command_1 = "cd " + working_directory
        #command_2 = "python " + working_directory + "randomTrips.py -n " + file + " -r " + file.replace(".net.xml", ".rou.xml") + " -e " + number + " l "

        #os.system('cmd \k' + command_1)
        #os.system('cmd \c' + str(command_2))

new_files = osmToXML()
