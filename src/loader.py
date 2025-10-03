from sklearn.datasets import load_iris
import pandas as pd 

"""
This function is made to load the iris dataset
"""
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    df['flower name'] = df.target.apply(lambda x : iris.target_names[x])

    return iris, df