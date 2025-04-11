import wandb
import copy
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Optional, Tuple

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from omegaconf import OmegaConf
from rich.logging import RichHandler 
from rich.progress import Progress
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from model import BFN
from utils_train import (
    seed_everything, log_cfg,
    checkpoint_training_state,
    init_checkpointing,
    log,
    update_ema,
    ddict,
    make_infinite,
    make_progress_bar, make_config, make_dataloaders, make_bfn,
)

from cadlib.cad_dataset import CADDataset
from cadlib.cad_transfer import get_output_sv
import torch.nn.functional as F
from networks.transformer import TextInputClassifier

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[
        RichHandler(rich_tracebacks=True, show_time=False),  # 控制台输出
    ],
)

logger = get_logger(__name__)

from concurrent.futures import ThreadPoolExecutor

def kl_loss(mean1, sigma1, mean2, sigma2):
    """
    Compute the KL divergence between two Gaussians with numerical stability.
    """

    tensor = next((x for x in (mean1, sigma1, mean2, sigma2) if isinstance(x, torch.Tensor)), None)
    assert tensor is not None, "At least one argument must be a Tensor."
    
    eps = 1e-6
    sigma1 = torch.nn.functional.softplus(sigma1) + eps if isinstance(sigma1, torch.Tensor) else torch.tensor(sigma1 + eps, device=tensor.device)
    sigma2 = torch.nn.functional.softplus(sigma2) + eps if isinstance(sigma2, torch.Tensor) else torch.tensor(sigma2 + eps, device=tensor.device)
    
    # 计算 KL 散度
    log_term = torch.log(sigma2) - torch.log(sigma1) / 2
    mean_diff_term = torch.square(mean1 - mean2)
    sigma_sq_term = sigma1
    return log_term + (sigma_sq_term + mean_diff_term) / (2 * torch.square(sigma2)) - 0.5

def mean_flat(tensor):
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

def setup_classifier(cfg) -> Tuple[nn.Module, dict, optim.Optimizer]:
    """Create the model, dataloader and optimizer"""
    dataloaders = make_dataloaders(cfg)
    model = TextInputClassifier(**cfg.model.parameters)
    print(model)
    params = model.parameters()
    optimizer = optim.AdamW(params=params, **cfg.optimizer)
    
    return model, dataloaders, optimizer

@torch.no_grad()
def validate(
        cfg,
        model: BFN,
        ema_model: nn.Module,
        val_dataloader: DataLoader,
        step: int,
        run: "neptune.Run",
        pbar: Optional[Progress],
        best_val_loss: float,
        checkpoint_root_dir: Optional[Path],
        accelerator: Accelerator,
) -> float:
    """Evaluate model on validation data and save checkpoint if loss improves"""
    dtype = {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[accelerator.mixed_precision]
    model_to_eval = ema_model if ema_model is not None else model
    model_to_eval.eval()
    pbar = pbar or Progress()
    max_steps = cfg.max_val_batches if cfg.max_val_batches > 0 else len(val_dataloader)
    val_id = pbar.add_task("Validating", visible=True, total=cfg.val_repeats * max_steps, transient=True, loss=math.nan)

    loss, mu_loss, sigma_loss, count = 0.0, 0.0, 0.0, 0
    for i in range(cfg.val_repeats):
        for idx, eval_batch in enumerate(val_dataloader):
            enabled = True if dtype in [torch.float16, torch.bfloat16] else False
            with torch.inference_mode(), torch.cuda.amp.autocast(dtype=dtype, enabled=enabled):
                theta = eval_batch["theta"].cuda().to(torch.float32)
                theta /= theta.sum(-1, keepdims=True)
                bs, seq_len, _ = theta.shape
                t = eval_batch["t_value"].cuda().unsqueeze(1).repeat(1, seq_len).to(torch.float32)
                
                mu = eval_batch["mu_label"][:, 0].unsqueeze(1).cuda()   
                sigma = eval_batch["sigma_label"][:, 1].unsqueeze(1).cuda()   
                
                out = model(theta, t)
                
                pred_mu = out[:, 0].unsqueeze(1)
                pred_sigma = out[:, 1].unsqueeze(1)                                             
                          
                loss_iter = kl_loss(mu, sigma, pred_mu, pred_sigma)
                loss_iter = mean_flat(loss_iter).mean().item()
                loss += loss_iter
                
                mu_loss += F.mse_loss(mu, pred_mu).item()  
                sigma_loss += F.mse_loss(sigma, pred_sigma).item()  
                
                count += 1      
                
            pbar.update(val_id, advance=1, loss=loss / count)
            if (idx + 1) >= max_steps:
                break
    loss /= count
    mu_loss /= count
    sigma_loss /= count
    pbar.remove_task(val_id)
    log(run["metrics"]["val"]["loss"], loss, step)
    
    if accelerator.is_main_process:
        wandb.log({"val_loss": loss, "val_mu_loss": mu_loss, "val_sigma_loss": sigma_loss})

    if checkpoint_root_dir is not None and (loss < best_val_loss or math.isinf(best_val_loss)):
        logger.info(f"loss improved: new value is {loss}")
        step_checkpoint_path = checkpoint_root_dir / "best"
        run_id = "BFN" if isinstance(run, defaultdict) else run["sys"]["id"].fetch()
        checkpoint_training_state(step_checkpoint_path, accelerator, ema_model, step, run_id)
        run["metrics/best/loss/metric"] = loss
        run["metrics/best/loss/step"] = step

    model.train()
    return loss


def train(
        cfg,
        accelerator: Accelerator, 
        model: BFN,
        ema_model: Optional[nn.Module],
        dataloaders: dict,
        optimizer: optim.Optimizer,
        run: "neptune.Run",
):
    is_main = accelerator.is_main_process
    pbar = make_progress_bar(is_main)
    run_id = "valid_t" if isinstance(run, defaultdict) else run["sys"]["id"].fetch()
    train_id = pbar.add_task(f"Training {run_id}", start=cfg.start_step, total=cfg.n_training_steps, loss=math.nan)
    
    checkpoint_root_dir = init_checkpointing(cfg.checkpoint_dir, run_id) if is_main else None
    best_val_loss = math.inf

    train_iter = make_infinite(dataloaders["train"])
    model.train()
    with pbar:
        tmp_loss = []
        for step in range(cfg.start_step, cfg.n_training_steps + 1):
            step_loss = 0.0
            for _ in range(cfg.accumulate):
                with accelerator.accumulate(model):
                    train_batch = next(train_iter)
                    theta = train_batch["theta"].cuda().to(torch.float32)
                    theta /= theta.sum(-1, keepdims=True)
                    bs, seq_len, _ = theta.shape
                    t = train_batch["t_value"].cuda().unsqueeze(1).repeat(1, seq_len).to(torch.float32)
                    
                    
                    mu = train_batch["mu_label"][:, 0].unsqueeze(1).cuda() 
                    sigma = train_batch["sigma_label"][:, 1].unsqueeze(1).cuda() 

                    out = model(theta, t)
                    
                    pred_mu = out[:, 0].unsqueeze(1)
                    pred_sigma = out[:, 1].unsqueeze(1)
                            
                    loss_iter = kl_loss(mu, sigma, pred_mu, pred_sigma)
                    loss_iter = mean_flat(loss_iter)
                    
                    loss = loss_iter.mean()
                                                    
                    mu_loss = F.mse_loss(mu, pred_mu).item()  
                    sigma_loss = F.mse_loss(sigma, pred_sigma).item() 
                                         
                    tmp_loss.append(loss.item())
                    accelerator.backward(loss)

                    if accelerator.sync_gradients and cfg.grad_clip_norm > 0:
                        accelerator.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                step_loss += loss.item()

            update_ema(ema_model, model, cfg.ema_decay)

            if is_main and (step % cfg.checkpoint_interval == 0):
                checkpoint_training_state(checkpoint_root_dir / "last", accelerator, ema_model, step, run_id)
                # run["checkpoints/last"].track_files(str(checkpoint_root_dir / "last"))               

            log(run["metrics"]["train"]["loss"], step_loss / cfg.accumulate, step, is_main and step % cfg.log_interval == 0)
            log(run["metrics"]["epoch"], step // len(dataloaders["train"]), step, is_main)
            
            wandb.log({"train_loss": step_loss / cfg.accumulate, "train_mu_loss": mu_loss, "train_sigma_loss": sigma_loss})

            if is_main and (step % cfg.val_interval == 0) and "val" in dataloaders:
                val_loss = validate(
                    cfg=cfg,
                    model=model,
                    ema_model=ema_model,
                    val_dataloader=dataloaders["val"],
                    step=step,
                    run=run,
                    pbar=pbar,
                    best_val_loss=best_val_loss,
                    checkpoint_root_dir=checkpoint_root_dir,
                    accelerator=accelerator,
                )
                best_val_loss = min(val_loss, best_val_loss)

            pbar.update(train_id, advance=1, loss=loss.item())
    

def main(cfg):
    acc = Accelerator(gradient_accumulation_steps=cfg.training.accumulate)

    seed_everything(cfg.training.seed)
    logger.info(f"Seeded everything with seed {cfg.training.seed}", main_process_only=True)

    with acc.main_process_first():
        model, dataloaders, optimizer = setup_classifier(cfg)
        
    if acc.is_main_process:
        wandb.init(project=cfg.meta.wandb_project)

    ema = copy.deepcopy(model) if acc.is_main_process and cfg.training.ema_decay > 0 else None  # EMA on main proc only
    model, optimizer, dataloaders["train"] = acc.prepare(model, optimizer, dataloaders["train"])
    run = ddict()
    if acc.is_main_process:
        ema.to(acc.device) 
        try:
            if cfg.meta.neptune:
                import neptune
                run = neptune.init_run(project=cfg.meta.neptune, mode="debug" if cfg.meta.debug else None)
                run["accelerate"] = dict(amp=acc.mixed_precision, nproc=acc.num_processes)
                log_cfg(cfg, run)
        except ImportError:
            logger.info("Did not find neptune installed. Logging will be disabled.")

    train(cfg.training, acc, model, ema, dataloaders, optimizer, run)

    if acc.is_main_process:
        wandb.finish()

if __name__ == "__main__":
    cfg = OmegaConf.load('./configs/cad_guided.yaml')
    cli_cfg = OmegaConf.from_cli()
    cfg_file = OmegaConf.merge(cfg, cli_cfg) 
    main(make_config(cfg_file))
