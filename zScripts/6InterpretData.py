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

    dataframes = []
    for fname in test_files:
        if fname == ".gitignore":
            pass
        else:
            data = pd.read_csv(os.path.join(data_dir, fname))
            dataframes.append(data)

    dijkstra = []
    dynamic_dijkstra = []
    static_local = []
    dynamic_local = []
    static_global = []
    dynamic_global = []
    for df in dataframes:
        p1 = df.sum(axis=0, skipna=True)
        p2 = df.sum(axis=1, skipna=True)
        p3 = df.sum(axis=2, skipna=True)
        p4 = df.sum(axis=3, skipna=True)
        total = p1 + p2 + p3 + p4
        name = df.name
        name = name.split("_")
        code = name[1]

        if code == "SDGC":
            static_global.append(total)
        elif code == "DDGC":
            dynamic_global.append(total)
        elif code == "SDLC":
            static_local.append(total)
        elif code == "DDLC":
            dynamic_local.append(total)
        elif code == "DD":
            dynamic_dijkstra.append(total)
        elif code == "D":
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
