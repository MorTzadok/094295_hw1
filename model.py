# Before using models, check:
#                   -there is no missing data
#                   -the data is normalized
#                   -categorical values are encoded
import sklearn as sk
import pandas as pd
import numpy
import csv
import os

def load_data(path):
    files = os.listdir(path)
    to_csv = []
    for file in range(0, len(files)):
        path_name = os.path.abspath(os.path.join(path,files[file]))
        to_csv.append(path_name)
    flag=True
    for i,one_file in enumerate(to_csv):
        with open(one_file, "r", encoding="utf-8") as f:
            if flag == True:
                patient_df = pd.read_csv(one_file,sep='|')
                patient_df['patient'] = files[i].split('patient')[1]
                flag = False
            else:
                new_patient_df = pd.read_csv(one_file, sep='|')
                new_patient_df['patient'] = files[i].split('patient')[1]
                patient_df = pd.concat([patient_df,new_patient_df], ignore_index = True)
    return patient_df.to_csv('patient_df.csv')


def train_models(train_data):

    y_train = train_data['SepsisLabel']
    X_train = train_data.drop(['SepsisLabel','patient'], axis = 1)

    #Training the models
    RForest = sk.ensemble.RandomForestClassifier(random_state=0,n_estimators=10, max_depth=None,min_samples_split=5)
    RForest.fit(X_train, y_train)

    KNN = sk.neighbors.KNeighborsClassifier(n_neighbors=3)
    KNN.fit(X_train, y_train)

    logReg = sk.linear_model.LogisticRegression
    logReg.fit(X_train, y_train)


    #try gradient boosting classifier?
    #We need to add grid search!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    return RForest,KNN,SGboost

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

    #calculate f1
    f1 = sk.metrics.f1_score(y_test, df_predictions['finalLabel'])
    print("The F1 score of the combined model on the test set is ",f1)

def main():
    #read train data
    path_train = 'data/train'
    load_data(path_train)
    path_test = 'data/test'
    # read test data
    load_data(path_test)

    #preProcessing
    ############## add uriel's part
    train_data = preProcessing()

    RForest,KNN,SGboost = train_models(train_data)
    classify_test(test_data,RForest,KNN,SGboost)


main()