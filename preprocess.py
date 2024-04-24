import os
from utils import GenSplit, DatasetAttentiveFP, createDataframe

split_set = 'train'

root = f'./data/{split_set}/data/'
config = './config/config.yml'

root_df = f'./data/{split_set}/data/raw/data/dataframe.pkl'
raw_data = './ORNL_data'

print("Creating dataframe...")
if not os.path.exists(root+'/raw/data/split_dict.pt'):
    num_molecules = createDataframe(root =root_df,
                raw_data = raw_data,
                config = config)
    if num_molecules is None:
        import pandas as pd
        df = pd.read_pickle(root_df)
        num_molecules = len(df)
        del df
    print("dataframe created...")
    print("Generating split...")
    GenSplit(root= root+'/raw/data/split_dict.pt',
            num_molecules = num_molecules,
            split=[0.94, 0.01, 0.05])
    print("split generated...")
print("Creating sqlite, dataset...")

root = f'./data/{split_set}/data/'
root_df = f'./data/{split_set}/data/raw/data/dataframe.pkl'
DatasetAttentiveFP(root=root, split=split_set, one_hot=True, config_path=config)

#root = './data/test/data/'
#root_df = './data/test/data/raw/data/dataframe.pkl'
#DatasetAttentiveFP(root=root, split='test', one_hot=True, config_path=config)


#root = './data/train/data/'
#root_df = './data/train/data/raw/data/dataframe.pkl'
#DatasetAttentiveFP(root=root, split='train', one_hot=True, config_path=config)


print("sqlite, dataset created...")
