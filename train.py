import pandas as pd
import numpy as np
import os
import warnings
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

os.chdir('C:/Users/danie/OneDrive - Instituto Tecnologico y de Estudios Superiores de Monterrey/ITESM/Machine Learning Bootcamp/')
SEED = 42

if __name__ == '__main__':
    np.random.seed(SEED)
    warnings.filterwarnings('ignore')

    # Load data
    data = pd.read_csv('data/Absenteeism_at_work.csv', sep=';')

    # Preprocessing data
    data.rename(columns = {'Work load Average/day ': 'Work load Average/day'}, inplace=True)
    data['y'] = np.where(data['Absenteeism time in hours'] <= 0, 1, 0)
    data.drop(columns=['Reason for absence', 'Absenteeism time in hours', 'ID'], inplace=True)
    
    # Split the data into training and testing sets
    train, test = train_test_split(data, train_size=0.8, random_state=SEED)

    # Prepare the data for training
    X_train = train.drop(columns=['y'])
    X_test = test.drop(columns=['y'])
    y_train = train[['y']]
    y_test = test[['y']]

    # Pipeline for the data
    num_pipeline = make_pipeline(SimpleImputer(strategy='median'), MinMaxScaler())
    num_columns = ['Transportation expense', 'Distance from Residence to Work', 'Service time', 'Age', 
                   'Work load Average/day', 'Hit target', 'Son', 'Pet', 'Weight', 'Height', 'Body mass index']

    cat_pipeline = make_pipeline(SimpleImputer(strategy='most_frequent'), OneHotEncoder(drop='first'))
    cat_columns = ['Month of absence', 'Day of the week', 'Seasons', 'Disciplinary failure', 
                   'Social drinker', 'Social smoker']
    
    ord_pipeline = make_pipeline(SimpleImputer(strategy='most_frequent'), OrdinalEncoder())
    ord_columns = ['Education']

    transform = make_column_transformer(
        (num_pipeline, num_columns),
        (cat_pipeline, cat_columns),
        (ord_pipeline, ord_columns),
        remainder='passthrough'
    )

    # Create the model
    rf = make_pipeline(transform, RandomForestClassifier(random_state=SEED))
    y_pred = cross_val_predict(rf, X_train, y_train.values.ravel(), cv=3, n_jobs=-1)

    # Evaluate the model
    accuracy = accuracy_score(y_train, y_pred)
    recall = recall_score(y_train, y_pred)
    precision = precision_score(y_train, y_pred)
    f1 = f1_score(y_train, y_pred)

    # Print out metrics
    print(f"Performance of random forest classifier in the validation set")
    print(f"  Accuracy: {accuracy}")
    print(f"  Recall: {recall}")
    print(f"  Precision: {precision}")
    print(f"  F1 Score: {f1}")