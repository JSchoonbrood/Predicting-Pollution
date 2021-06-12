import os, sys, csv
from pathlib import Path
from sys import platform


current_dir = Path(os.path.dirname(__file__))
if sys.platform == "linux" or sys.platform == "linux2":
    data_dir = os.path.join(current_dir.parent, 'CSV/')
    new_dir = os.path.join(current_dir.parent, 'PROCESSED_CSV/')
elif sys.platform == "win32":
    data_dir =  os.path.join(current_dir.parent, 'CSV\\')
    new_dir = os.path.join(current_dir.parent, 'PROCESSED_CSV\\')

data_files = os.listdir(data_dir)

for file_name in data_files:
    processed_file_name = new_dir + str(file_name.replace(".csv", "_processed.csv"))
    with open(processed_file_name, 'w', newline='') as processed_file:
        writer = csv.writer(processed_file)
        with open(os.path.join(data_dir, file_name), 'r', newline='') as file:
            reader = csv.reader(file)
            data = list(reader)
            for row in data:
                if (row[1] == row[2] == row[3] == row[4] == row[5]) and (int(row[8]) >= 0):
                    pass
                else:
                    writer.writerow(row)
        file.close()
    processed_file.close()
