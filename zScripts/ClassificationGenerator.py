import os, sys, csv
import math
import statistics
import numpy
from pathlib import Path
from sklearn.preprocessing import LabelBinarizer

current_dir = Path(os.path.dirname(__file__))
if sys.platform == "linux" or sys.platform == "linux2":
    processed_dir = os.path.join(current_dir.parent, 'PROCESSED_CSV/')
    ranked_dir = os.path.join(current_dir.parent, 'RANKED_CSV/')
elif sys.platform == "win32":
    processed_dir =  os.path.join(current_dir.parent, 'PROCESSED_CSV\\')
    ranked_dir =  os.path.join(current_dir.parent, 'RANKED_CSV\\')

ranks = [1, 2, 3, 4, 5, 6, 7, 8]
rank_values = []
for fname in (os.listdir(processed_dir)):
    ranked_file_name = os.path.join(ranked_dir, (str(fname.replace("_processed.csv", "_ranked.csv"))))
    with open(ranked_file_name, 'w', newline='') as ranked_file:
        writer = csv.writer(ranked_file)
        with open(os.path.join(processed_dir, fname), 'r', newline='') as file:
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
                    basic_rank = math.floor(basic_rank/60)
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
