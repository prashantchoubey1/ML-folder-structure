import pandas as pd
import config 
from sklearn.model_selection import KFold

# Reading training dataframe
df_train = pd.read_csv(config.TRAINING_FILE)
# Create a fold column with default value -1
df_train['fold']=-1
# Fixing the number of folds required in the data
k=5
# Creating object of KFold class
kf = KFold(n_splits=k, random_state=1, shuffle=True)
# Pushing the value of folds in each split
for i,(arr1_,arr2_) in enumerate(kf.split(X=df_train)):
    df_train.loc[arr2_,'fold'] = i
# Rewriting the dataframe back to excel for further manipulation
df_train.to_csv(config.TRAINING_FOLD_FILE,reset_index=False)
