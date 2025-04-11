import os 
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

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[
        RichHandler(rich_tracebacks=True, show_time=False), 
    ],
)

logger = get_logger(__name__)

from concurrent.futures import ThreadPoolExecutor

def setup(cfg) -> Tuple[nn.Module, dict, optim.Optimizer]:
    """Create the model, dataloader and optimizer"""
    dataloaders = make_dataloaders(cfg)
    model = make_bfn(cfg.model)
    print(model)
    if "weight_decay" in cfg.optimizer.keys() and hasattr(model.net, "get_optim_groups"):
        params = model.net.get_optim_groups(cfg.optimizer.weight_decay)
    else:
        params = model.net.parameters()
    # Instantiate the optimizer using the hyper-parameters in the config
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

    loss, mse, count, valid_count_all = 0.0, 0.0, 0, 0
    for i in range(cfg.val_repeats):
        for idx, eval_batch in enumerate(val_dataloader):
            enabled = True if dtype in [torch.float16, torch.bfloat16] else False
            with torch.inference_mode(), torch.cuda.amp.autocast(dtype=dtype, enabled=enabled):
                for k, _ in eval_batch.items():
                    eval_batch[k] = eval_batch[k].cuda()
                loss += model_to_eval(eval_batch).item()
                count += 1
                
                samples = model_to_eval.sample(eval_batch["cad_vec"].shape, 100, eval_batch["conditions"])
                output_sv, valid_count = get_output_sv(samples)
                output_sv = torch.tensor(output_sv, dtype=torch.float32).to(samples.device)
                
                valid_count_all += valid_count

                if valid_count == 0:
                    mse += 0.0
                else:
                    
                    mask = output_sv != -1
                    output_sv = output_sv[mask]
                    conditions = eval_batch["conditions"][mask]                
                    
                    mse += F.mse_loss(conditions, output_sv).item()               
                
            pbar.update(val_id, advance=1, loss=loss / count)
            if (idx + 1) >= max_steps:
                break
    loss /= count
    mse /= count
    valid_pi = valid_count_all / 500
    pbar.remove_task(val_id)
    log(run["metrics"]["val"]["loss"], loss, step)
    log(run["metrics"]["val"]["mse"], mse, step)
    
    # 记录验证损失到 WandB
    if accelerator.is_main_process:
        wandb.log({"val_loss": loss, "val_mse": mse, "valid_pi": valid_pi})

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
    run_id = "BFN" if isinstance(run, defaultdict) else run["sys"]["id"].fetch()
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

                    loss = model(train_batch)
                    tmp_loss.append(loss.item())
                    # print(tmp_loss)
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
            
            wandb.log({"train_loss": step_loss / cfg.accumulate})

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
        model, dataloaders, optimizer = setup(cfg)

    if acc.is_main_process:
        wandb.init(project=cfg.meta.wandb_project)

    ema = copy.deepcopy(model) if acc.is_main_process and cfg.training.ema_decay > 0 else None  # EMA on main proc only
    model, optimizer, dataloaders["train"] = acc.prepare(model, optimizer, dataloaders["train"])
    
    acc.load_state("checkpoints/model_area/BFN/last")  
    
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
    cfg = OmegaConf.load('./configs/cad_discrete.yaml')
    cli_cfg = OmegaConf.from_cli()
    cfg_file = OmegaConf.merge(cfg, cli_cfg) 
    main(make_config(cfg_file))
