import torch
from omegaconf import OmegaConf, DictConfig
# from data import QuoraDataset
from tqdm import tqdm  
from utils_train import seed_everything, make_config, make_bfn
import re
import torch
import numpy as np
from torch.utils.data import DataLoader
from cadlib.cad_dataset import CADDataset
from cadlib.cad_transfer import vec2sv, vec2step, save_steps
import torch.nn.functional as F
from networks.transformer import TextInputClassifier

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True

def pearson_correlation(x, y): 
    x = np.array(x)
    y = np.array(y)
    
    n = len(x)
    numerator = n * np.sum(x * y) - np.sum(x) * np.sum(y)
    denominator = np.sqrt((n * np.sum(x**2) - np.sum(x)**2) * (n * np.sum(y**2) - np.sum(y)**2))
    
    if denominator == 0:
        return 0 
    else:
        return numerator / denominator 

def main(cfg: DictConfig) -> torch.Tensor:
    """
    Config entries:
        seed (int): Optional                                                                             
        config_file (str): Name of config file containing model and data config for a saved checkpoint
        load_model (str): Path to a saved checkpoint to be tested
        sample_shape (list): Shape of sample batch
        n_steps (int): Number of sampling steps (positive integer).
        save_file (str): File path to save the generated sample tensor. Skip saving if None.
    """
    seed_everything(cfg.seed)
    print(f"Seeded everything with seed {cfg.seed}")

    # Get model config from the training config file
    bfn_cfg = make_config(cfg.bfn_config)
    bfn = make_bfn(bfn_cfg.model)
    bfn.load_state_dict(torch.load(cfg.load_model, weights_only=True, map_location="cpu"))
    
    guided_cfg = make_config(cfg.guided_config)
    guided_model = TextInputClassifier(**guided_cfg.model.parameters)
    print(guided_model)

    guided_model.load_state_dict(torch.load(cfg.guided_model, weights_only=True, map_location="cpu"))
    
    if torch.cuda.is_available():
        bfn.to("cuda")
        guided_model.to("cuda")

    cad_dataset = CADDataset("test", bfn_cfg.data)
    dataloader = DataLoader(cad_dataset, batch_size=100, shuffle=False)
            
    valid_count, all_count, count, area_mse, vol_mse, area_mae, vol_mae, person_area, person_vol = 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    
    for batch in tqdm(dataloader, desc="Processing batches"):
        count += 1
        
        cad_vec = batch['cad_vec'].clone().detach().cpu().to(torch.int32).numpy()
        conditions = batch['conditions'].clone().detach().to("cuda")                
        init_conditions = batch['init_conditions'].clone().detach().to("cuda")

        with torch.no_grad():
            samples = bfn.sample(cad_vec.shape, cfg.n_steps, conditions, init_conditions, guided_model).cpu().to(torch.int32).numpy()
            # samples = bfn.sample(cad_vec.shape, cfg.n_steps, conditions)
           
        all_count += conditions.shape[0]        
        conditions = init_conditions
    
        output_sv = []
        mat = samples    
        
        for output_vec in mat:
            # print(output_vec)
            try: 
                area, vol = vec2sv(output_vec, is_mat=False)
            except:
                area, vol = -1, -1
            if not area == vol == -1:
                valid_count += 1
            output_sv.append([area, vol])    
        output_sv = torch.tensor(output_sv, dtype=torch.float32).to(conditions.device)

        area_mask = output_sv[:, 0] != -1
        vol_mask = output_sv[:, 1] != -1

        gt_area, pred_area = conditions[area_mask, 0], output_sv[area_mask, 0]
        gt_vol, pred_vol = conditions[vol_mask, 1], output_sv[vol_mask, 1]

        gt_area_cpu, pred_area_cpu = gt_area.cpu(), pred_area.cpu()
        gt_vol_cpu, pred_vol_cpu = gt_vol.cpu(), pred_vol.cpu()

        person_area += pearson_correlation(gt_area_cpu, pred_area_cpu)
        person_vol += pearson_correlation(gt_vol_cpu, pred_vol_cpu)

        area_mse_val = F.mse_loss(gt_area, pred_area).item()
        vol_mse_val = F.mse_loss(gt_vol, pred_vol).item()
        area_mae_val = F.l1_loss(gt_area, pred_area).item()
        vol_mae_val = F.l1_loss(gt_vol, pred_vol).item()

        area_mse += area_mse_val
        vol_mse += vol_mse_val
        area_mae += area_mae_val
        vol_mae += vol_mae_val

    valid_pi = valid_count / all_count
    area_mse /= count
    vol_mse /= count
    area_mae /= count
    vol_mae /= count
    person_area /= count
    person_vol /= count
        
    print(f"\nResults:")
    print("-" * 30)
    print(f"Valid Proportion (valid_pi): {valid_pi:.4f}")
    print(f"Area Mean Squared Error (area_mse): {area_mse:.4f}")
    print(f"Volume Mean Squared Error (vol_mse): {vol_mse:.4f}")
    print(f"Area Mean Absolute Error (area_mae): {area_mae:.4f}")
    print(f"Volume Mean Absolute Error (vol_mae): {vol_mae:.4f}")
    print(f"person_area: {person_area:.4f}")
    print(f"person_vol: {person_vol:.4f}")
    print("-" * 30)
    
if __name__ == "__main__":
    import time
    
    cfg = OmegaConf.load('./configs/config.yaml')
    cli_cfg = OmegaConf.from_cli()
    cfg = OmegaConf.merge(cfg, cli_cfg) 
    
    start_time = time.time()

    print("starting.....")
    
    main(cfg)
    
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    print(f"执行时间: {elapsed_time:.2f} 秒")