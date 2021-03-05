import os, sys, csv
import math
import statistics
import numpy
from sklearn.preprocessing import LabelBinarizer

working_directory = os.getcwd() + r"\\SUMO\\"
data_files = os.listdir(working_directory + r"\\PROCESSED_CSV\\")

ranks = [1, 2, 3, 4, 5, 6, 7, 8]
rank_values = []
for file_name in data_files:
    ranked_file_name = str(working_directory + r"RANKED_CSV\\") + str(file_name.replace("_processed.csv", "_ranked.csv"))
    with open(ranked_file_name, 'w', newline='') as ranked_file:
        writer = csv.writer(ranked_file)
        with open((working_directory + r"PROCESSED_CSV\\" + file_name), 'r', newline='') as file:
            reader = csv.reader(file)
            data = list(reader)
            step = 0
            for row in data:
                if step == 0:
                    step += 1
                    new_row = ["Mean_Speed", "Estimated_Travel_Time", "Traffic_Level", "Total_Neighbours", "Length", "Rank"]#, "Targets"]
                    #row.append("ohe")
                    writer.writerow(new_row)
                else:
                    basic_rank = ((1*float(row[1])) + (1*float(row[2])) + (1*float(row[3])) + (1*float(row[4])))/float(row[9])
                    basic_rank = math.floor(basic_rank/160)
                    if basic_rank > 7:
                        basic_rank = 7

                    rank = ranks[basic_rank]

                    new_row = [row[5], row[6], row[7], row[8], row[9], rank]

                    writer.writerow(new_row)

                    # CODE BELOW TO TEST OHE WORKING CORRECTLY (INSPECT CSV FILES)
                    #rank_values.append(new_row)
        #onehot_encoder = LabelBinarizer(sparse_output=False)
        #test = []
        #for row in rank_values:
        #    test.append(row[int(-1)])
        #labels = onehot_encoder.fit_transform(test)

        #index = 0
        #for row in rank_values:
        #    row.append(labels[index])
        #    writer.writerow(row)
        #    index += 1

        #rank_values = []

        file.close()
    ranked_file.close()
