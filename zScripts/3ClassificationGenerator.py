import os
import sys
import csv
import math
from pathlib import Path

current_dir = Path(os.path.dirname(__file__))
if sys.platform == "linux" or sys.platform == "linux2":
    processed_dir = os.path.join(current_dir.parent, 'PROCESSED_CSV/')
    ranked_dir = os.path.join(current_dir.parent, 'RANKED_CSV/')
elif sys.platform == "win32":
    processed_dir = os.path.join(current_dir.parent, 'PROCESSED_CSV\\')
    ranked_dir = os.path.join(current_dir.parent, 'RANKED_CSV\\')

ranks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
rank_values = []
for fname in (os.listdir(processed_dir)):
    if fname != ".gitignore":
        ranked_file_name = os.path.join(
                                        ranked_dir,
                                        (str(fname.replace("_processed.csv",
                                         "_ranked.csv"))))
        with open(ranked_file_name, 'w', newline='') as ranked_file:
            writer = csv.writer(ranked_file)

            with open(os.path.join(processed_dir, fname), 'r', newline='') as file:
                reader = csv.reader(file)
                data = list(reader)
                step = 0
                for row in data:
                    if step == 0:
                        step += 1
                        new_row = ["Mean_Speed", "Estimated_Travel_Time",
                                   "Traffic_Level", "Total_Neighbours",
                                   "Length", "OverallRank"]
                        writer.writerow(new_row)
                    else:
                        basic_rank = (((1*float(row[2]))
                                      + (100*float(row[3])) + (100*float(row[4]))
                                      + (1000*float(row[5])) / (10000)))

                        rank_index = math.floor(basic_rank/350)

                        if rank_index >= 12:
                            rank_index = 11

                        rank = ranks[rank_index]

                        new_row = [row[5], row[6], row[7], row[8], row[9],
                                   rank]

                        writer.writerow(new_row)

            file.close()
        ranked_file.close()
