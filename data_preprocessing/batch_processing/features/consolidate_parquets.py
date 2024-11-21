import pandas as pd
import os
import sys
from tqdm import tqdm


input_dir = sys.argv[1]
output_filename = sys.argv[2]

filelist = os.listdir(input_dir)
filelist = [filename for filename in filelist if "astro_objects_batch" in filename]

df_list = []
for filename in tqdm(filelist):
    df = pd.read_parquet(
        os.path.join(input_dir, filename))
    df_list.append(df)

df = pd.concat(df_list, axis=0)
df.to_parquet(output_filename)
