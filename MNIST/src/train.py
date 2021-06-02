import pandas as pd
from sklearn import tree 
from sklearn import metrics
import joblib
import config
import os
import argparse

df_train = pd.read_csv(config.TRAINING_FOLD_FILE)

def run_predictions(k):
    # Creating training and test sets based on the folds
    df_train_fold = df_train[df_train['fold']!=k].reset_index(drop=True)
    df_test_fold = df_train[df_train['fold']==k].reset_index(drop=True)

    # Creating training and test sets separating feature and labels
    X_train = df_train_fold.drop(columns='label')
    X_test = df_train_fold.label.values
    y_train = df_test_fold.drop(columns='label')
    y_test = df_test_fold.label.values

    # Create a model object
    classifier = 'decision_tree'
    dt_model = tree.DecisionTreeClassifier()
    dt_model.fit(X_train,X_test)
    y_pred = dt_model.predict(y_train)

    # Accuracy calculation
    print('SCORE FOR FOLD = ',k)
    print('Accuracy score is ',round(metrics.accuracy_score(y_pred,y_test)*100,2))
    print('Macro F1 score is ',round(metrics.f1_score(y_pred,y_test,average='macro')*100,2))
    print('Micro F1 score is ',round(metrics.f1_score(y_pred,y_test,average='micro')*100,2))
    print('Weighted F1 score is ',round(metrics.f1_score(y_pred,y_test,average='weighted')*100,2),'\n')

    # Saving the model
    joblib.dump(dt_model, os.path.join(config.MODEL_OUTPUT,f"{classifier}_{k}_folds.bin"))

if __name__== '__main__':
    # Initializing arhument parser to take input from command line
    parser = argparse.ArgumentParser()
    # Add agrument with the name and type for user input
    parser.add_argument('--folds',type=int)
    # Read the argument from command line
    arg = parser.parse_args()
    # Pass the argument read from command line to the function
    run_predictions(arg.folds)