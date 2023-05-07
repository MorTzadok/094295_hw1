# Before using models, check:
#                   -there is no missing data
#                   -the data is normalized
#                   -categorical values are encoded
import sklearn as sk
import pandas as pd
import numpy
import csv
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
####################!!!??
def load_data(path):
    print(f'_{path.split("/")[1]}')
    files = os.listdir(path)
    to_csv = []
    for file in range(0, len(files)):
        path_name = os.path.abspath(os.path.join(path,files[file]))
        to_csv.append(path_name)
    flag=True
    for i,one_file in enumerate(to_csv):
      print(i)
      with open(one_file, "r", encoding="utf-8") as f:
          if flag == True:
              patient_df = pd.read_csv(one_file,sep='|')
              patient_df['patient'] = files[i].split('patient')[1]
              flag = False
          else:
              new_patient_df = pd.read_csv(one_file, sep='|')
              new_patient_df['patient'] = files[i].split('patient')[1]
              patient_df = pd.concat([patient_df,new_patient_df], ignore_index = True)
    str_save = f'patient_df_{path.split("/")[1]}_1.csv'
    return patient_df.to_csv(str_save)
    

def train_models(train_data):

    y_train = train_data['SepsisLabel']
    X_train = train_data.drop(['SepsisLabel','patient'], axis = 1)

    #Training the models
    RForest = RandomForestClassifier(random_state=0,n_estimators=10, max_depth=50,min_samples_split=5)
    RForest.fit(X_train, y_train)

    KNN = KNeighborsClassifier(n_neighbors=3)
    KNN.fit(X_train, y_train)

    logReg = LogisticRegression(random_state=0, C=5)
    logReg.fit(X_train, y_train)


    #try gradient boosting classifier?
    #We need to add grid search!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    return RForest,KNN,logReg

def classify_test(test_data,RForest,KNN,logReg):
    #evaluate the model
    y_test = test_data['SepsisLabel']
    X_test = test_data.drop(['SepsisLabel', 'patient'], axis=1)
    df_predictions = pd.DataFrame({})

    #predict Labels for test set
    df_predictions['RForest'] = RForest.predict(X_test)
    f1_rf = sk.metrics.f1_score(y_test, df_predictions['RForest'])
    print('The F1 score of the random forest on the test set is ',f1_rf)

    df_predictions['KNN'] = KNN.predict(X_test)
    f1_knn = sk.metrics.f1_score(y_test, df_predictions['KNN'])
    print('The F1 score of the KNN on the test set is ',f1_knn)

    df_predictions['logReg'] = logReg.predict(X_test)
    f1_lr = sk.metrics.f1_score(y_test, df_predictions['logReg'])
    print('The F1 score of the logistic regression model on the test set is ',f1_lr)


    #Choose the label that is the output of at list 2 of the models
    df_predictions['finalLabel'] = df_predictions.mode(axis=1)
    df_predictions.to_csv('/home/student/094295_hw1/predictions.csv')
    #calculate f1
    f1 = sk.metrics.f1_score(y_test, df_predictions['finalLabel'])
    print("The F1 score of the combined model on the test set is ",f1)
    
    
def preProcessing(df):
    print(df['patient'])
    df['patient'] = df['patient'].apply(lambda x: x.split("_")[1].split(".")[0])
    df = df.sort_values(['patient', 'ICULOS'])
    groups = df.groupby('patient')

    def filter_group(group):
        # Find the index of the first row with a "1" in the binary column
        group_1 = group.loc[group['SepsisLabel'] == 1]
    
        if group_1.empty:
            # Find the row with the least number of nulls
            least_nulls_row = group.loc[group.iloc[:, 0] == group.isnull().sum(axis=1).idxmin()]
    
            # Calculate the row-wise average of all the other rows
            average_row = group.mean(skipna=True)
    
            # Fill null values in the least-nulls row with the average row values
            filled_row = least_nulls_row.fillna(average_row)
            return filled_row
    
        idx = group_1['SepsisLabel'].idxmax()
    
        selected_row = group_1.loc[idx].fillna(method='ffill')
    
        # Return only the selected row, and drop the rest of the rows
        return selected_row.to_frame().T

    # Apply the custom function to each group, and concatenate the results
    filtered_df = pd.concat([filter_group(group) for _, group in groups])
    return filtered_df.fillna(df.mean())
    
def main():
    #read train data
    path_train = 'data/train'
    #load_data(path_train)
    path_test = 'data/test'
    # read test data
    #load_data(path_test)

    #preProcessing

#    df_train = pd.read_csv("/home/student/094295_hw1/patient_df_train_1.csv")
#    df_test = pd.read_csv("/home/student/094295_hw1/patient_df_test_1.csv")
#    train_data = preProcessing(df_train)
#    test_data = preProcessing(df_test)
#    train_data.to_csv('/home/student/094295_hw1/train_processed.csv')
#    test_data.to_csv('/home/student/094295_hw1/test_processed.csv')
#    print("finished pre-processing")

    train_data = pd.read_csv("/home/student/094295_hw1/train_processed.csv")
    test_data = pd.read_csv("/home/student/094295_hw1/test_processed.csv")
    RForest,KNN,logReg = train_models(train_data)
    classify_test(test_data,RForest,KNN,logReg)


main()