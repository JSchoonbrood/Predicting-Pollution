import os
import sys

working_directory = r"C:\\Users\\Jake\\Desktop\\Projects\\SUMO\\OSM\\TRAINING_SIMS\\SUMO_FILES\\"
sys.path.insert(1, working_directory)
import randomTrips
#os.system('cmd /k "Your Command Prompt Command"')


def osmToXML():
    working_directory = r"C:\\Users\\Jake\\Desktop\\Projects\\SUMO\\OSM\\TRAINING_SIMS\\"
    osm_files = os.listdir(working_directory)
    for fname in osm_files:
        path = os.path.join(working_directory, fname)
        if os.path.isdir(path):
            continue

        path = os.path.join(working_directory, fname)
        new_path = os.path.join((working_directory + "SUMO_FILES\\"), fname)
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
            file.write('<end value = "2000"/>\n')
            file.write('</time>\n\n')
            file.write('</configuration>\n')
            file.flush()
        file.close()

    return osm_files



def ranTripGen(osm_files):
    number = "3000"
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
