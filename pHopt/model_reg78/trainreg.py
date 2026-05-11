import warnings
warnings.filterwarnings('ignore')
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32,garbage_collection_threshold:0.6"
import argparse
import random
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from modelreg import *
from tqdm import tqdm
import time
from sklearn.metrics import ( mean_squared_error, mean_absolute_error, r2_score)
from scipy.stats import pearsonr, spearmanr
from transformers import get_cosine_schedule_with_warmup
from dataset import load_embeddings, data_load_seq, vocab_size



def seed_everything(seed=42):
 
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def performance(y_true_reg, y_pred_reg):
  
    
    y_true_reg = y_true_reg.flatten()
    y_pred_reg = y_pred_reg.flatten()
    r2 = r2_score(y_true_reg, y_pred_reg)
    rmse = np.sqrt(mean_squared_error(y_true_reg, y_pred_reg))
    mae = mean_absolute_error(y_true_reg, y_pred_reg)
    try:
        pearson = pearsonr(y_true_reg, y_pred_reg)[0]
    except:
        pearson = 0.0
    try:
        spearman = spearmanr(y_true_reg, y_pred_reg)[0]
    except:
        spearman = 0.0

    # Concordance Index
    def ci(y_true, y_pred):
        n = 0
        h_sum = 0.0
        for i in range(len(y_true)):
            for j in range(i+1, len(y_true)):
                if y_true[i] != y_true[j]:
                    n += 1
                    h_sum += ((y_pred[i]-y_pred[j])*(y_true[i]-y_true[j]) > 0)
        return h_sum/n if n>0 else 0.0

    ci_value = ci(y_true_reg, y_pred_reg)

    metrics = {
        "R2": round(r2,4),
        "Pearson": round(pearson,4),
        "Spearman": round(spearman,4),
        "RMSE": round(rmse,4),
        "MAE": round(mae,4),
        "CI": round(ci_value,4)
    }

    return metrics



def write_logfile(epoch, train_loss, valid_loss, train_metrics, valid_metrics, logfile):
   
    columns = [
        "epoch", 
        "train_loss", 
        "train_r2", "train_pearson", "train_spearman", "train_rmse", "train_mae", "train_ci",
        "valid_loss", 
        "valid_r2", "valid_pearson", "valid_spearman", "valid_rmse", "valid_mae", "valid_ci"
    ]

    values = [
        epoch,
        train_loss,
        train_metrics.get("R2", np.nan),
        train_metrics.get("Pearson", np.nan),
        train_metrics.get("Spearman", np.nan),
        train_metrics.get("RMSE", np.nan),
        train_metrics.get("MAE", np.nan),
        train_metrics.get("CI", np.nan),
        valid_loss,
        valid_metrics.get("R2", np.nan),
        valid_metrics.get("Pearson", np.nan),
        valid_metrics.get("Spearman", np.nan),
        valid_metrics.get("RMSE", np.nan),
        valid_metrics.get("MAE", np.nan),
        valid_metrics.get("CI", np.nan)
    ]

    df = pd.DataFrame([values], columns=columns)

    if not os.path.exists(logfile):
        df.to_csv(logfile, index=False, float_format="%.4f")
    else:
        df.to_csv(logfile, mode='a', header=False, index=False, float_format="%.4f")


def log_epoch_results(epoch, total_epochs, start_time, train_loss, valid_loss, train_metrics, valid_metrics):
   

    elapsed = time.time() - start_time
    print(
        f"\nEpoch [{epoch}/{total_epochs}]: "
        f"Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f} | "
        f"Train R2: {train_metrics.get('R2',0):.4f}, RMSE: {train_metrics.get('RMSE',0):.4f} | "
        f"Valid R2: {valid_metrics.get('R2',0):.4f}, RMSE: {valid_metrics.get('RMSE',0):.4f} | "
        f"Time: {elapsed:.2f}s"
    )



def build_optimizer_scheduler(models, base_lr=2e-5, weight_decay=1e-5,scheduler_type="plateau",
                              mode="max", factor=0.5, patience=10, verbose=True):


    param_groups = []
    for name, model in models.items():
        lr = base_lr
        param_groups.append({"params": model.parameters(), "lr": lr, "weight_decay": weight_decay})


    optimizer = optim.AdamW(param_groups)


    if scheduler_type.lower() == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=mode, factor=factor, patience=patience, verbose=verbose
        )
    elif scheduler_type.lower() == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=patience, eta_min=1e-6
        )
    else:
        scheduler = None

    return optimizer, scheduler


def load_optimizer_checkpoint(optimizer, path, strict=True):
    
    checkpoint = torch.load(path, map_location="cpu")
    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    else:
        print("[Warning] 'optimizer_state_dict' not found in checkpoint")
    return optimizer



class EarlyStopping:
    
    def __init__(self, espatience=40, verbose=True, path='checkpoint.pt', monitor="R2", mode="max"):
       
        self.espatience = espatience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.path = path
        self.monitor = monitor

        
        if mode is None:
            if monitor.lower() in ["r2", "pearson", "spearman", "ci"]:
                self.mode = "max"
            else:  # RMSE, loss, etc.
                self.mode = "min"
        else:
            self.mode = mode

    def __call__(self, metrics_dict, model, optimizer=None, scheduler=None):
       

        score = metrics_dict.get(self.monitor, None)
        if score is None:
            raise ValueError(f"Metric '{self.monitor}' not found in metrics_dict. Available options: {list(metrics_dict.keys())}")

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, optimizer, scheduler)
            if self.verbose:
                print(f"📈 Initial best {self.monitor}: {self.best_score:.4f}")
            return
        
        
        improved = (score > self.best_score) if self.mode == "max" else (score < self.best_score)

        if improved:
            self.best_score = score
            self.save_checkpoint(model, optimizer, scheduler)
            self.counter = 0
            if self.verbose:
                print(f"📈 New best {self.monitor}: {self.best_score:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"⏱ {self.monitor} did not improve for {self.counter}/{self.espatience} epochs")
            if self.counter >= self.espatience:
                self.early_stop = True
                if self.verbose:
                    print(f"⛔ Early stopping triggered. Best {self.monitor}: {self.best_score:.4f}")
        

    def save_checkpoint(self, model, optimizer=None, scheduler=None):
        
        if self.verbose:
            print(f"📦 Saving model checkpoint to {self.path}")
        checkpoint = {
            "model_state_dict": model.state_dict()
        }
        if optimizer is not None:
            try:
                checkpoint["optimizer_state_dict"] = optimizer.state_dict()
            except Exception:
                checkpoint["optimizer_state_dict"] = None
        if scheduler is not None:
            try:
                checkpoint["scheduler_state_dict"] = scheduler.state_dict()
            except Exception:
                checkpoint["scheduler_state_dict"] = None

        torch.save(checkpoint, self.path)



def prepare_training_independent(
    save_dir,
    model_name,
    build_model_fn,
    optimizer_fn=None,     
    scheduler_fn=None,
    restart=False,
    dataloaders=None,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
):
    
    savepath = os.path.join(save_dir, model_name)
    os.makedirs(savepath, exist_ok=True)
    recent_model = os.path.join(savepath, "model_recent.pt")
    best_model = os.path.join(savepath, "model_best.pt")
    log_file = os.path.join(savepath, "log.csv")

    
    model = build_model_fn()
    model.to(device)

    optimizer = optimizer_fn(model) if optimizer_fn is not None else None
    scheduler = scheduler_fn(optimizer) if (scheduler_fn is not None and optimizer is not None) else None

    
    if not restart:
        progress_params = {
            'epoch': 0,
            'best_val_r2': -np.inf, 
        }
        print("🆕 Starting training from scratch.")
    else:
        
        assert os.path.exists(log_file), "Cannot resume: log_file not found"
        

        logs = pd.read_csv(log_file)
        last_row = logs.iloc[-1]
        last_epoch = int(last_row.get('epoch', 0))
        best_val_r2 = logs['valid_r2'].max()

        
        progress_params = {
            'epoch': last_epoch+1,       
            'best_val_r2': float(best_val_r2),
        }

        
        if os.path.exists(recent_model):
            checkpoint = torch.load(recent_model, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"🔄 Resuming training from epoch {progress_params['epoch']} using recent_model.pt")
            else:
                model.load_state_dict(checkpoint) 
        
         
            if optimizer is not None and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            elif optimizer is not None:
                print("⚠️ optimizer state not found in checkpoint, initializing new optimizer.")

            
            if scheduler is not None and 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            elif scheduler is not None:
                print("⚠️ scheduler state not found in checkpoint, initializing new scheduler.")
        else:
            print("⚠️ recent_model.pt not found, starting from scratch.")
    
    paths = {
        'savepath': savepath,
        'recent_model': recent_model,
        'best_model': best_model,
        'log_file': log_file
    }

    return model, optimizer, scheduler, progress_params, paths


def train_one_epoch(model, dataloader, optimizer, device, reg_loss_func, 
                    epoch_idx=None, total_epochs=None, use_amp=False, 
                    scheduler=None, accumulation_steps=1):


    model.train()
    total_loss_samples = 0.0
    total_samples = 0
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    all_reg_true = []
    all_reg_pred = []

    pbar = tqdm(dataloader, desc=f"Epoch {epoch_idx}/{total_epochs}" if epoch_idx is not None else "Training")

    
    optimizer.zero_grad()
    
    
    for i, batch in enumerate(pbar):
        seq_inputs, reg_labels, esm_embeddings, esm_mask, weight, ids = batch
        seq_inputs = seq_inputs.to(device)           # [B, L]
        reg_labels = reg_labels.to(device).squeeze() # [B]
        weight = weight.to(device).squeeze() 
        if esm_embeddings is not None:
            esm_embeddings = esm_embeddings.to(device)
        if esm_mask is not None:
            esm_mask = esm_mask.to(device)


        with torch.cuda.amp.autocast(enabled=use_amp):

            pred_reg, fusion, att = model(seq_inputs, esm_embeddings, esm_mask)  


            reg_loss = reg_loss_func(pred_reg, reg_labels)
            
            loss = reg_loss / accumulation_steps


        scaler.scale(loss).backward()

       
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dataloader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            scaler.step(optimizer)
            scaler.update()

          
            if scheduler is not None:
                scheduler.step()
                
            optimizer.zero_grad()
       
        loss_val = loss.item() * accumulation_steps
        batch_size = seq_inputs.size(0)
        total_loss_samples += loss.item() * batch_size       
        total_samples += batch_size


        all_reg_true.append(reg_labels.detach().cpu().numpy())
        all_reg_pred.append(pred_reg.detach().cpu().numpy())

       
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix({"loss": f"{loss_val:.4f}", "lr": f"{current_lr:.2e}"})

    avg_loss = total_loss_samples / total_samples

    all_reg_true = np.concatenate(all_reg_true, axis=0)
    all_reg_pred = np.concatenate(all_reg_pred, axis=0)

    metrics_dict = performance(all_reg_true+ 7.0, all_reg_pred+ 7.0)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return avg_loss, metrics_dict



def valid_one_epoch(model, dataloader, device, reg_loss_func, use_amp=False, linear_correction=True):
 

    model.eval() 
    total_loss_samples = 0.0
    total_samples = 0

    all_reg_true = []
    all_reg_pred = []


   
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validating")

        for batch in pbar:
            seq_inputs, reg_labels, esm_embeddings, esm_mask, weight, ids = batch
            seq_inputs = seq_inputs.to(device)
            reg_labels = reg_labels.to(device).squeeze()
            weight = weight.to(device).squeeze() 
            if esm_embeddings is not None:
                esm_embeddings = esm_embeddings.to(device)
            if esm_mask is not None:
                esm_mask = esm_mask.to(device)

            with torch.cuda.amp.autocast(enabled=use_amp):
                
                pred_reg,fusion,att = model(seq_inputs, esm_embeddings, esm_mask)  


                reg_loss = reg_loss_func(pred_reg, reg_labels)

                
                loss = reg_loss

            batch_size = seq_inputs.size(0)
            total_loss_samples += loss.item() * batch_size
            total_samples += batch_size


            all_reg_true.append(reg_labels.detach().cpu().numpy())
            all_reg_pred.append(pred_reg.detach().cpu().numpy())

            pbar.set_postfix({"val_loss": f"{loss.item():.4f}"})

    avg_loss = total_loss_samples / total_samples


    all_reg_true = np.concatenate(all_reg_true, axis=0)
    all_reg_pred = np.concatenate(all_reg_pred, axis=0)



    real_true = all_reg_true + 7.0
    real_pred = all_reg_pred + 7.0
    metrics_dict = performance(real_true, real_pred)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return avg_loss, metrics_dict




def parse_args():
    parser = argparse.ArgumentParser(description='Train Enzyme Regression Model')
    
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    default_csv = os.path.join(current_dir, '../../data/pHopt/pHopt_7_8.csv')
    default_esm = os.path.join(current_dir, '../../data/pHopt/esm_embedding')
    default_save = os.path.join(current_dir, '../../train_model/pHopt/savetrain')
    
   
    parser.add_argument('--csv_path', default=default_csv, type=str)
    parser.add_argument('--esm_path', default=default_esm, type=str)
    parser.add_argument('--save_dir', default=default_save, type=str)
    
    
    parser.add_argument('--seq_max_len', default=1024, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--espatience', default=18, type=int)
    parser.add_argument('--lr', default=2e-5, type=float)
    parser.add_argument('--weight_decay', default=1e-2, type=float)
    parser.add_argument('--accumulation_steps', default=4, type=int)
    return parser.parse_args()



def main():


    seed_everything(42)
    args = parse_args()
    esm_features = load_embeddings(args.esm_path, max_length=args.seq_max_len)
    train_loader = data_load_seq(csv_path=args.csv_path, 
                                 batch_size=args.batch_size, 
                                 split='Training', 
                                 esm_loader=esm_features,
                                 seq_max_len=args.seq_max_len)
    valid_loader = data_load_seq(
        csv_path=args.csv_path, batch_size=args.batch_size * 2, split='Validation', 
        esm_loader=esm_features, seq_max_len=args.seq_max_len
    )

    reg_loss_func = nn.MSELoss() 

    model = Mymodel_phreg1(vocab_size=vocab_size).to(device)

    
   
    num_training_steps = int(len(train_loader) * args.epochs / args.accumulation_steps)
    num_warmup_steps = int(num_training_steps * 0.05) 
    print(f"🔥 Steps: Total={num_training_steps}, Warmup={num_warmup_steps}")

   
    def optimizer_fn(model):

        return optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def scheduler_fn(optimizer):
        return get_cosine_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=num_warmup_steps, 
            num_training_steps=num_training_steps
        )


    model, optimizer, scheduler, progress, paths = prepare_training_independent(
        args.save_dir, model_name="phreg78", build_model_fn=lambda: model,
        optimizer_fn=optimizer_fn, scheduler_fn=scheduler_fn,
        restart=False, device=device
    )


    # --- EarlyStopping ---
    early_stopper = EarlyStopping(
        espatience=args.espatience, verbose=True, path=paths['best_model'],
        monitor="R2", mode="max")


    start_epoch = progress['epoch']
    best_val_r2 = progress['best_val_r2']

    print(f"🚀 Start Training from Epoch {start_epoch} / Max Epochs = {args.epochs}")

    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()

        
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, optimizer, device, reg_loss_func,
            epoch_idx=epoch+1, total_epochs=args.epochs,
            use_amp=True,
            scheduler=scheduler,  
            accumulation_steps=args.accumulation_steps 
        )

        
        valid_loss, valid_metrics = valid_one_epoch(
            model, valid_loader, device, reg_loss_func)


        early_stopper(valid_metrics, model, optimizer, scheduler)

        if early_stopper.best_score is not None:
            best_val_r2 = early_stopper.best_score

        
        write_logfile(epoch, train_loss, valid_loss, train_metrics, valid_metrics, paths['log_file'])
        log_epoch_results(epoch, args.epochs, start_time, train_loss, valid_loss, train_metrics, valid_metrics)

        
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None
        }, paths['recent_model'])
        print(f"💾 Saved recent_model for epoch {epoch}")

        
        import gc
        torch.cuda.empty_cache()   
        gc.collect()               

        if early_stopper.early_stop:
            print(f"⛔ Early stopping at epoch {epoch}")
            break

    print(f"\n🎯 Training finished. Best validation R2 = {best_val_r2:.4f}")
    print(f"📦 Best model saved to: {paths['best_model']}")

if __name__ == "__main__":
    main()

