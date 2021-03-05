import os, sys, csv

working_directory = os.getcwd() + r"\\SUMO\\"
data_files = os.listdir(working_directory + r"\\CSV\\")

for file_name in data_files:
    processed_file_name = str(working_directory + r"PROCESSED_CSV\\") + str(file_name.replace(".csv", "_processed.csv"))
    with open(processed_file_name, 'w', newline='') as processed_file:
        print (processed_file_name)
        writer = csv.writer(processed_file)
        print ((working_directory + r"CSV\\" + file_name))
        with open((working_directory + r"CSV\\" + file_name), 'r', newline='') as file:
            reader = csv.reader(file)
            data = list(reader)
            for row in data:
                if (row[1] == row[2] == row[3] == row[4]) and (int(row[7]) >= 0):
                    pass
                else:
                    writer.writerow(row)
        file.close()
    processed_file.close()
