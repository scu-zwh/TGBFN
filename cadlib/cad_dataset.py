from torch.utils.data import Dataset, DataLoader
import torch
import os
import json
import h5py
import random
import sys
from cadlib.macro import *

def matrix_to_vector(matrix):
    flattened_array = matrix.flatten()
    result_vector = flattened_array[flattened_array != -1]
    return result_vector

def normalize_data(sv_data):
    normalization_param = {'area': [np.float64(2.314415928650784), np.float64(2.054904572083571)],
                            'vol': [np.float64(0.1710946203777644), np.float64(0.26940128764260857)]}      
    return {
        "area": (sv_data["area"] - normalization_param["area"][0]) / normalization_param["area"][1],
        "vol": (sv_data["vol"] - normalization_param["vol"][0]) / normalization_param["vol"][1]
    }  

class CADDataset(Dataset):
    def __init__(self, phase, config):
        super(CADDataset, self).__init__()
        self.raw_data = os.path.join(config.data_root, "cad_vec") # h5 data root
        self.phase = phase
        self.aug = config.augment
        self.path = os.path.join(config.data_root, "my_train_val_test_split.json")
        self.sv_path = os.path.join(config.data_root, "sv_data.json")
        with open(self.path, "r") as fp:
            self.all_data = json.load(fp)[phase]
        with open(self.sv_path, "r") as fp:
            self.all_sv_data = json.load(fp)[phase]

        self.seq_len = config.seq_len
        self.all_data_lengths = []
        print(f"Loading {len(self.all_data)} data entries. Filtering in progress...")
        print(f"{phase} dataset has {len(self.all_data)} data.")
        
        self.normalization_param = {'area': [np.float64(2.314415928650784), np.float64(2.054904572083571)],
                                    'vol': [np.float64(0.1710946203777644), np.float64(0.26940128764260857)]}
        
    def get_data_by_id(self, data_id):
        idx = self.all_data.index(data_id)
        return self.__getitem__(idx)
    
    def normalize_data(self, sv_data):
        return {
            "area": (sv_data["area"] - self.normalization_param["area"][0]) / self.normalization_param["area"][1],
            "vol": (sv_data["vol"] - self.normalization_param["vol"][0]) / self.normalization_param["vol"][1]
        }     
    
    def __getitem__(self, index):
        data_id = self.all_data[index]
        h5_path = os.path.join(self.raw_data, data_id + ".h5")
        sv_data = self.all_sv_data[data_id]
        
        init_conditions = np.array([sv_data["area"], sv_data["vol"]], dtype=float)
        
        sv_data = self.normalize_data(sv_data)
        with h5py.File(h5_path, "r") as fp:
            cad_mat = fp["vec"][:] # (len, 1 + N_ARGS)
        cad_mat[:, 0] += 256
        cad_vec = matrix_to_vector(cad_mat)
            
        cad_vec = np.pad(cad_vec, (0, self.seq_len-len(cad_vec)), mode='constant',
                         constant_values=PAD_IDX)
        
        conditions = np.array([sv_data["area"], sv_data["vol"]], dtype=float)
        
        cad_vec = torch.tensor(cad_vec, dtype=torch.int32) 
        conditions = torch.tensor(conditions, dtype=torch.float32)
        init_conditions = torch.tensor(init_conditions, dtype=torch.float32)

        return {"cad_vec": cad_vec, "conditions": conditions,
                "init_conditions": init_conditions, "id": data_id}

    def __len__(self):
        return len(self.all_data)

if __name__ == "__main__":
    from omegaconf import OmegaConf
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    cfg = OmegaConf.load('configs/cad_discrete.yaml')
    
    cad_dataset = CADDataset("train", cfg.data)
    print(cad_dataset)

    data_loader = DataLoader(cad_dataset, batch_size=100, shuffle=False, num_workers=8)

    for i, batch in enumerate(tqdm(data_loader)):
        if batch is None:  
            continue    