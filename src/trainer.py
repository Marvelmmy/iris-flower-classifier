from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import pickle

""" This function is made to train the model with RandomForestClassifier """

def train_model(df, modelfile="my_model.pkl"):
    X = df.drop(df[['target', 'flower name']], axis='columns')
    y = df['target']

    # splitting the dataset (train/test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)

    # Defining the Model
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    # Save model via pickle
    with open(modelfile, 'wb') as f:
        pickle.dump(model, f)
    
    return model, X_test, y_test