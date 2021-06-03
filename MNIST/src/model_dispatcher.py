# Description : This file is just to create a dictionary of models so that it could be imported
# in another files and can be directly implemented using keys
from sklearn import ensemble,tree

model = {
    'rf' : ensemble.RandomForestClassifier(n_jobs=-1),
    'dt' : tree.DecisionTreeClassifier()
}