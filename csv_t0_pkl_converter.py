import numpy as np
import pandas as pd

csv_file_path = "data/nodes_seq_feat.csv"
df = pd.read_csv(csv_file_path)
pickle_file_path = 'seqFeature.pkl'
# Save the DataFrame to a pickle file
df.to_pickle(pickle_file_path)

print(f"CSV file '{csv_file_path}' converted to pickle file '{pickle_file_path}'")