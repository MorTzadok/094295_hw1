import pandas as pd
import numpy
import csv
import os


files = os.listdir('data/train')
to_csv = []
for file in range(0, len(files)):
    path_name = os.path.abspath(os.path.join('data/train',files[file]))
    to_csv.append(path_name)
flag=True
for i,one_file in enumerate(to_csv):
    with open(one_file, "r", encoding="utf-8") as f:
        if flag == True:
            patient_df = pd.read_csv(one_file,sep='|')
            patient_df['patient'] = str(i)
            flag = False
        else:
            new_patient_df = pd.read_csv(one_file, sep='|')
            new_patient_df['patient'] = str(i)
            patient_df = pd.concat([patient_df,new_patient_df], ignore_index = True)
patient_df.to_csv('patient_df.csv')


