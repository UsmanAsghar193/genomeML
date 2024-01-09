import pandas as pd


pickle_file_path = 'seqFeature.pkl'

# Loading the pickle file into a pandas DataFrame
df = pd.read_pickle(pickle_file_path)

print(df)