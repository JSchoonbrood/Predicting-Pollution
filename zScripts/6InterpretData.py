import numpy as np
import pandas as pd
import os
import sys

from matplotlib import pyplot
from pathlib import Path

def run():
    current_dir = Path(os.path.dirname(__file__))
    if sys.platform == "linux" or sys.platform == "linux2":
        data_dir = os.path.join(current_dir.parent, 'TEST_CSV/')
    elif sys.platform == "win32":
        data_dir = os.path.join(current_dir.parent, 'TEST_CSV\\')

    test_files = os.listdir(data_dir)
    print (test_files)
    for i in range(len(test_files)):
        if test_files[i] == "map_1_DDGC_13.csv":
            del test_files[i]
            break


    dataframes = []
    for fname in test_files:
        if fname == ".gitignore":
            pass
        else:
            data = pd.read_csv(os.path.join(data_dir, fname))

            name = fname
            name = name.split("_")
            code = name[2]

            if code == "SDGC":
                data.name = "SDGC"
            elif code == "DDGC":
                data.name = "DDGC"
            elif code == "SDLC":
                data.name = "SDLC"
            elif code == "DDLC":
                data.name = "DDLC"
            elif code == "DD":
                data.name = "DD"
            elif code == "D":
                data.name = "D"

            dataframes.append(data)

    dijkstra = []
    dynamic_dijkstra = []
    static_local = []
    dynamic_local = []
    static_global = []
    dynamic_global = []
    for df in dataframes:
        p1 = df.sum(axis=1, skipna=True)
        total = 0
        for i in p1:
            total += i
        name = df.name

        if name == "SDGC":
            static_global.append(total)
        elif name == "DDGC":
            dynamic_global.append(total)
        elif name == "SDLC":
            static_local.append(total)
        elif name == "DDLC":
            dynamic_local.append(total)
        elif name == "DD":
            dynamic_dijkstra.append(total)
        elif name == "D":
            dijkstra.append(total)

    names = ["Dij", "Dyn_Dij", "Stat_Dij_Local", "Dyn_Dij_Local", "Stat_Dij_Global", "Dyn_Dij_Global"]
    index = 0
    for i in [dijkstra, dynamic_dijkstra, static_local, dynamic_local, static_global, dynamic_global]:
        data = i
        sum = 0
        for l in range(len(data)):
            sum += data[l]
        average = sum/len(data)
        print (names[index], average)
        index += 1

run()