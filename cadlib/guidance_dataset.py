import os
import torch
import h5py

from torch.utils.data import Dataset, DataLoader

class GuidanceDataset(Dataset):
    def __init__(self, phase, config):
        self.directory = os.path.join(config.data_root, phase) 
        self.files = os.listdir(self.directory)
        
        print(f"load {len(self.files)} samples")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.directory, self.files[idx])
        with h5py.File(file_path, 'r') as h5_file:
            theta = h5_file['theta'][:]  
            label = h5_file['label'][:] 
            mu_label = h5_file['mu_label'][:]  
            sigma_label = h5_file['sigma_label'][:]  
            t_value = h5_file['t_value'][()] 
            
        dict_data = {
            "theta": theta,
            "label": label,
            "mu_label": mu_label,
            "sigma_label": sigma_label,
            "t_value": t_value
        }
        
        return dict_data

if __name__ == "__main__":
    from omegaconf import OmegaConf
    from tqdm import tqdm
    
    cfg_file = "configs/cad_guided.yaml"
    cfg = OmegaConf.load(cfg_file)

    dataset = GuidanceDataset("train", cfg.data)
   
    data_loader = DataLoader(dataset, batch_size=100, shuffle=False, num_workers=8)

    for i, batch in enumerate(tqdm(data_loader)):

        print(i)


    
        
        # dict_data = batch
        # print("Batch theta shape:", dict_data["theta"].shape)
        # print("Batch theta shape:", dict_data["label"].shape)