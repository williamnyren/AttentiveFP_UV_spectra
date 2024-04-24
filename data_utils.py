import os
import os.path as osp
from typing import Any, Callable, Dict, List, Optional

import torch
from tqdm import tqdm

from torch_geometric.data import Data, OnDiskDataset, download_url, extract_zip
from torch_geometric.data.data import BaseData, Data
from torch_geometric_modified import from_smiles

import yaml

import pandas as pd
import numpy as np
import multiprocessing as mp
from functools import partial


class DatasetAttentiveFP(OnDiskDataset):
    r"""The PCQM4Mv2 dataset from the `"OGB-LSC: A Large-Scale Challenge for
    Machine Learning on Graphs" <https://arxiv.org/abs/2103.09430>`_ paper.
    :class:`PCQM4Mv2` is a quantum chemistry dataset originally curated under
    the `PubChemQC project
    <https://pubs.acs.org/doi/10.1021/acs.jcim.7b00083>`_.
    The task is to predict the DFT-calculated HOMO-LUMO energy gap of molecules
    given their 2D molecular graphs.

    .. note::
        This dataset uses the :class:`OnDiskDataset` base class to load data
        dynamically from disk.

    Args:
        root (str): Root directory where the dataset should be saved.
        split (str, optional): If :obj:`"train"`, loads the training dataset.
            If :obj:`"val"`, loads the validation dataset.
            If :obj:`"test"`, loads the test dataset.
            If :obj:`"holdout"`, loads the holdout dataset.
            (default: :obj:`"train"`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        backend (str): The :class:`Database` backend to use.
            (default: :obj:`"sqlite"`)
    """
    #url = ('https://dgl-data.s3-accelerate.amazonaws.com/dataset/OGB-LSC/'
    #       'pcqm4m-v2.zip')

    split_mapping = {
        'train': 'train',
        'val': 'valid',
        'test': 'test',
    }

    def __init__(
        self,
        root: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        backend: str = 'sqlite',
        one_hot = False,
        config = None
    ) -> None:
        assert split in ['train', 'test', 'val']
        self.split = split
        self.one_hot = one_hot
        self.config = config
        if config is None:
            print('Please provide the configuration file path. Exiting...')
            return
                
        if self.one_hot:
            schema = {
                'x': dict(dtype=torch.float, size=(-1, 26)),
                'edge_index': dict(dtype=torch.long, size=(2, -1)),
                'edge_attr': dict(dtype=torch.float, size=(-1, 14)),
                'smiles': str,
                'y': dict(dtype=torch.float, size=(-1, self.config['out_dim'])),
            }
        else:
            schema = {
                'x': dict(dtype=torch.float, size=(-1, 9)),
                'edge_index': dict(dtype=torch.long, size=(2, -1)),
                'edge_attr': dict(dtype=torch.float, size=(-1, 3)),
                'smiles': str,
                'y': dict(dtype=torch.float, size=(-1, self.config['out_dim'])),
            }

        super().__init__(root, transform, backend=backend, schema=schema)

        split_idx = torch.load(self.raw_paths[1])
        self._indices = split_idx[self.split_mapping[split]].tolist()

    
    
    @property
    def raw_file_names(self) -> List[str]:
        #print(f"In raw_file_names: {self._indices[0:10]}")
        return [
            osp.join('data', 'dataframe.pkl'),
            osp.join('data', 'split_dict.pt'),
        ]
    def get_indecies(self):
        return self._indices

    def download(self) -> None:
        pass        
    
    def process(self) -> None:
        #print(f"In process: {self._indices[0:10]}")
        # Process all at once. Then use self._indices to select data.
        import pandas as pd
        import time
        split_idx = torch.load(self.raw_paths[1])
        indices = split_idx[self.split_mapping[self.split]].tolist()
        df = pd.read_pickle(self.raw_paths[0])
        df = df.iloc[indices]  
        data_list: List[Data] = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            data = from_smiles(row['smiles'], with_hydrogen=True, one_hot=self.one_hot)
            y = torch.tensor(row['smooth_spectra'], dtype=torch.float).view(-1, self.config['out_dim'])
            data.y = y
            data_list.append(data)
            if i + 1 == len(df) or (i + 1) % 1000 == 0:  # Write batch-wise:
                self.extend(data_list)
                data_list = []

    def serialize(self, data: BaseData) -> Dict[str, Any]:
        assert isinstance(data, Data)
        #print(f"In serialize: {self._indices[0:10]}")
        return dict(
            x=data.x,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            y=data.y,
            smiles=data.smiles,
        )

    def deserialize(self, data: Dict[str, Any]) -> Data:
        #print(f"In deserialize: split: {self.split}, {self._indices[0:5]}")
        return Data.from_dict(data)

class GenSplit():
    r""" Args:
            root (str): Root directory where the dataset should be saved.
            num_molecules (int): Number of molecules in the dataset.
            split (list): List of floats representing the fraction of the dataset
                that should be used for training, validation, and test. The sum
                of the list elements should be 1. (default: [0.8, 0.1, 0.1])
        
    """
    def __init__(self, root = 'split_dict.pt', num_molecules = 10582578 ,split = [0.8, 0.1, 0.1], force_recreate = False):
        if osp.exists(root) and not force_recreate:
            print('Split file already exists. Skipping...')
            return
        else:
            os.makedirs(osp.dirname(root), exist_ok=True)
        self.num_molecules = num_molecules-1
        self.split = split
        assert sum(split) == 1
        self.indices = torch.arange(num_molecules)
        split_dict = {'train': self.indices[0:int(num_molecules*split[0])],
                      'valid': self.indices[int(num_molecules*split[0]):int(num_molecules*(split[0]+split[1]))],
                      'test': self.indices[int(num_molecules*(split[0]+split[1])):]}
        torch.save(split_dict, root)

# Function to find SMILES with '.' in them. '.' splits the molecule into multiple molecules
def search_for_dots(smis):
    df_lst_reject = []
    df_lst_pass = []
    for smi in smis:
        if '.' in smi:
            df_lst_reject.append(smi)
        else:
            df_lst_pass.append(smi)
        
    return df_lst_reject, df_lst_pass

def smooth_spectra(config, df_split):
    def convert_ev_in_nm(ev_value):
        planck_constant = 4.1357 * 1e-15  # eV s
        light_speed = 299792458  # m / s
        meter_to_nanometer_conversion = 1e+9
        return 1 / ev_value * planck_constant * light_speed * meter_to_nanometer_conversion
    
    def energy_to_wavelength(ev_list, prob_list):
        nm_list = [convert_ev_in_nm(value) for value in ev_list]
        combined = list(zip(nm_list, prob_list))
        sorted_combined = sorted(combined, key=lambda x: x[0])
        nm_list, prob_list = zip(*sorted_combined)
        return nm_list, prob_list

    def gauss(a, m, x, w):
        # calculation of the Gaussian line shape
        # a = amplitude (max y, intensity)
        # x = position
        # m = maximum/median (stick position in x, wave number)
        # w = line width, FWHM
        return a * np.exp(-(np.log(2) * ((m - x) / w) ** 2))
    
    def raw_to_smooth(nm_list, prob_list, config, w = 10.0):
        spectrum_discretization_step = config['resolution']
        xmin_spectrum = 0
        xmax_spectrum = config['max_wavelength']
        xmax_spectrum_tmp = xmax_spectrum*2

        gauss_sum = list()  # list for the sum of single gaussian spectra = the convoluted spectrum

        # plotrange must start at 0 for peak detection
        x = np.arange(xmin_spectrum, xmax_spectrum_tmp, spectrum_discretization_step)

        # plot single gauss function for every frequency freq
        # generate summation of single gauss functions
        for index, wn in enumerate(nm_list):
            # sum of gauss functions
            gauss_sum.append(gauss(prob_list[index], x, wn, w))

        # y values of the gauss summation
        gauss_sum_y = np.sum(gauss_sum, axis=0)
        #gauss_sum_y = gauss_sum_y / np.sum(gauss_sum_y)
        
        # find the index of x = encoder['min_wavelength'] in x
        index_min = int(config['min_wavelength']/spectrum_discretization_step)
        
        x = x[index_min:int(len(x)/2)]
        gauss_sum_y = gauss_sum_y[index_min:int(len(gauss_sum_y)/2)]
        #gauss_sum_y = gauss_sum_y / np.max(gauss_sum_y)

        xdata = x
        ydata = gauss_sum_y
        xlimits = [np.min(xdata), np.max(xdata)]

        y = []
        for elements in range(len(xdata)):
            if xlimits[0] <= xdata[elements] <= xlimits[1]:
                y.append(ydata[elements])
        y = np.array(y)
        y_max = np.max(y)
        y = y/y_max
        y = np.where(y < 1e-3, 0, y)
        y = y * y_max
        return np.array(y)
    ex_indices = [i for i in range(len(df_split.columns)) if df_split.columns[i][:2] == 'ex']
    prob_indices = [i for i in range(len(df_split.columns)) if df_split.columns[i][:4] == 'prob']
    df_smooth = {}
    df_smooth['smiles'] = []
    df_smooth['smooth_spectra'] = []
    for df in df_split.iterrows():
        df = df[1]
        ev_list = df.iloc[ex_indices].values
        prob_list = df.iloc[prob_indices].values
        nm_list, prob_list = energy_to_wavelength(ev_list, prob_list)
        y = raw_to_smooth(nm_list, prob_list, config, config['peak_width'])
        df_smooth['smiles'].append(df['smiles'])
        df_smooth['smooth_spectra'].append(y)
    
    # Dataframe with SMILES and the smoothed spectra
    df_smooth = pd.DataFrame(df_smooth)
    return df_smooth

def createDataframe(root: str = '', raw_data: str = '',
                    config: str = '', num_cpus: int = 0,
                    force_recreate: bool = False):
    import pandas as pd
    if os.path.exists(root) and not force_recreate:
        print('Dataframe already exists. Skipping...')
        return
    if raw_data == '':
        print('Please provide the raw data file path. Exiting...')
        return
    else:
        if not os.path.exists(raw_data):
            print('Raw data file not found. Exiting...')
            return
    if config == '':
        print('Please provide the configuration file path. Exiting...')
        return
    else:
        if not os.path.exists(config):
            print('Configuration file not found. Exiting...')
            return
    if root == '':
        print('Please provide the root directory to store dataframe. Exiting...')
        return
    else:
        dirs = root.split('/')
        dirs = os.path.join(*dirs[:-1])
        if not os.path.exists(dirs):
            print(f'Root directory not found. Creating...' + dirs)
            os.makedirs(dirs, exist_ok=True)
            
    if num_cpus  == 0:
        num_cpus = max(os.cpu_count()-2, 2)
    else:
        num_cpus = 2
        
    with open(config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = config['parameters']
    config['resolution'] = (config['max_wavelength']
                            - config['min_wavelength']) / config['out_dim']
    
    def filter_smiles(raw_data):
        raw_data_tmp = os.path.join(raw_data, 'post_data')
        df_small = pd.read_csv(raw_data_tmp + '/gdb9_ex.csv')
        # List all files in the directory with names './../ORNL_data/post_data/ornl_aisd_ex_*.csv'
        files = os.listdir(raw_data_tmp)
        files = [f for f in files if 'ornl_aisd_ex_' in f]
        # Read all files into a list of dataframes
        dfs_large = [pd.read_csv(raw_data_tmp+ '/' + f) for f in files]
        #Concatinate all dataframes into one
        df_large = pd.concat(dfs_large)
        df_all = pd.concat([df_small, df_large])
        # List all SMILES strings in the post data
        smiles_large = df_large['smiles'].values
        smiles_small = df_small['smiles'].values
        smiles_all = np.concatenate((smiles_large, smiles_small))
        smiles_all = np.array_split(smiles_all, num_cpus)
        
        # Create a pool of worker processes
        pool = mp.Pool(num_cpus)

        # Use partial function to pass the check_smiles function and the smiles_post list to the map function
        func = partial(search_for_dots)
        results = pool.map(func, smiles_all)
        # Close the pool and wait for the work to finish
        pool.close()
        pool.join()

        # Make a dataframe from the results
        df_res_reject = []
        df_res_pass = []
        for df_res in results:
            df_res_reject += df_res[0]
            df_res_pass += df_res[1]
            
        df_reject = pd.DataFrame(df_res_reject, columns=['smiles'])
        df_pass = pd.DataFrame(df_res_pass, columns=['smiles'])

        # Remove the rows with '.' in the smiles
        df_all = df_all[~df_all['smiles'].isin(df_reject['smiles'])]
        # Shuffle the dataframe
        df_all = df_all.sample(frac=1).reset_index(drop=True)
        # Save the dataframe to pkl
        df_all.to_pickle(raw_data + '/df_filtered.pkl')
        return df_all
    
    if not os.path.exists(raw_data + '/df_filtered.pkl') or force_recreate:
        df_all = filter_smiles(raw_data)

    else:
        df_all = pd.read_pickle(raw_data + '/df_filtered.pkl')

    if len(df_all) > 1000000:
        n = 200
        length = len(df_all)
        size = length // n
        if length % n != 0:
            size += 1
        df_all = [df_all[i * size:(i + 1) * size] for i in range(n)]
        
    elif len(df_all) > 100000:
        n = 20
        length = len(df_all)
        size = length // n
        if length % n != 0:
            size += 1
        df_all = [df_all[i * size:(i + 1) * size] for i in range(n)]
    else:
        n = num_cpus
        length = len(df_all)
        size = length // n
        if length % n != 0:
            size += 1
        df_all = [df_all[i * size:(i + 1) * size] for i in range(n)]
    
    # Create a pool of worker processes
    pool = mp.Pool(num_cpus)

    # Use partial function to pass the check_smiles function and the smiles_post list to the map function
    func = partial(smooth_spectra, config)
    results = pool.map(func, df_all)

    # Close the pool and wait for the work to finish
    pool.close()
    pool.join()

    results = pd.concat(results)

    dataframe_dir = root.split('/')
    dataframe_dir = os.path.join(*dataframe_dir[:-1])
    os.makedirs(dataframe_dir + '/', exist_ok=True)
    
    results.to_pickle(root)
    print(results.head(2))
    
    return len(results)
    print('Dataframe created and saved to {}'.format(root)) 
    
    
    