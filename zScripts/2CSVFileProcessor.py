import os, sys, csv
from pathlib import Path
from sys import platform

# Sets Up Directories
current_dir = Path(os.path.dirname(__file__))
if sys.platform == "linux" or sys.platform == "linux2":
    data_dir = os.path.join(current_dir.parent, 'CSV/')
    new_dir = os.path.join(current_dir.parent, 'PROCESSED_CSV/')
elif sys.platform == "win32":
    data_dir =  os.path.join(current_dir.parent, 'CSV\\')
    new_dir = os.path.join(current_dir.parent, 'PROCESSED_CSV\\')

# Pre-Processed Datafiles
data_files = os.listdir(data_dir)

for fname in data_files:
    processed_fname = (new_dir + str(fname.replace(".csv", "_processed.csv")))
    with open(processed_fname, 'w', newline='') as processed_file:
        writer = csv.writer(processed_file)
        with open(os.path.join(data_dir, file_name), 'r', newline='') as file:
            reader = csv.reader(file)
            data = list(reader)
            for row in data:
                # Row (1-5): Emissions, Row (8): No. Of Vehicles
                if (row[1] == row[2] == row[3] == row[4] == row[5] and
                    int(row[8]) >= 0):
                    pass
                else:
                    writer.writerow(row)
        file.close()
    processed_file.close()
