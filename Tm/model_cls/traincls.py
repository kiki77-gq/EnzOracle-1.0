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
from modelcls import *
from tqdm import tqdm
import time
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score)
from dataset import load_embeddings, cls_data_load_seq, vocab_size




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


def performance(y_true_cls, y_pred_cls_prob, threshold=0.5):

  
    if isinstance(y_pred_cls_prob, torch.Tensor):
        y_pred_cls_prob = y_pred_cls_prob.detach().cpu().numpy()
    if isinstance(y_true_cls, torch.Tensor):
        y_true_cls = y_true_cls.detach().cpu().numpy()

   
    y_pred_cls = (y_pred_cls_prob >= threshold).astype(int) 

    accuracy = accuracy_score(y_true_cls, y_pred_cls)
    precision = precision_score(y_true_cls, y_pred_cls, zero_division=0)
    recall = recall_score(y_true_cls, y_pred_cls, zero_division=0)
    f1 = f1_score(y_true_cls, y_pred_cls, zero_division=0)

    metrics = {
        "Accuracy": round(accuracy, 4),
        "F1": round(f1, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4)
    }

    return metrics



def write_logfile(epoch, train_loss, valid_loss, train_metrics, valid_metrics, logfile):
  
    columns = [
        "epoch", 
        "train_loss", "train_accuracy", "train_f1", "train_precision", "train_recall",
        "valid_loss", "valid_accuracy", "valid_f1", "valid_precision", "valid_recall"
    ]

    values = [
        epoch,
        train_loss,
        train_metrics.get("Accuracy", np.nan),
        train_metrics.get("F1", np.nan),
        train_metrics.get("Precision", np.nan),
        train_metrics.get("Recall", np.nan),
        valid_loss,
        valid_metrics.get("Accuracy", np.nan),
        valid_metrics.get("F1", np.nan),
        valid_metrics.get("Precision", np.nan),
        valid_metrics.get("Recall", np.nan)
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
        f"Train Accuracy: {train_metrics.get('Accuracy',0):.4f} | "
        f"Valid Accuracy: {valid_metrics.get('Accuracy',0):.4f} | "
        f"Time: {elapsed:.2f}s"
    )



def build_optimizer_scheduler(models, base_lr=1e-5, weight_decay=1e-5,scheduler_type="plateau",
                              mode="max", factor=0.8, patience=10, verbose=True):


   
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
  
    def __init__(self, espatience=40, verbose=True, path='checkpoint.pt', monitor="Accuracy", mode="max"):
     
        self.espatience = espatience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.path = path
        self.monitor = monitor
        self.mode = "max"


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
        
     
        improved = score > self.best_score

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
            'best_val_accuracy': -np.inf, 
        }
        print("🆕 Starting training from scratch.")
    else:
        
        assert os.path.exists(log_file), "Cannot resume: log_file not found"
        

        logs = pd.read_csv(log_file)
        last_row = logs.iloc[-1]
        last_epoch = int(last_row.get('epoch', 0))
        best_val_accuracy = logs['valid_accuracy'].max()

        
        progress_params = {
            'epoch': last_epoch+1,       
            'best_val_accuracy': float(best_val_accuracy),
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


def train_one_epoch(model, dataloader, optimizer, device, cls_loss_func,  
                    epoch_idx=None, total_epochs=None, use_amp=False):
  

    model.train()
    total_loss_samples = 0.0
    total_samples = 0
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    all_cls_true = []
    all_cls_hard_true = []
    all_cls_pred_prob = []


    pbar = tqdm(dataloader, desc=f"Epoch {epoch_idx}/{total_epochs}" if epoch_idx is not None else "Training")

    for batch in pbar:
        seq_inputs, reg_labels, cls_labels, esm_embeddings, esm_mask, weight, ids = batch
        seq_inputs = seq_inputs.to(device)           # [B, L]
        cls_labels = cls_labels.to(device) # [B]
        weight = weight.to(device).squeeze()
        if esm_embeddings is not None:
            esm_embeddings = esm_embeddings.to(device)
        if esm_mask is not None:
            esm_mask = esm_mask.to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=use_amp):

            cls_probs, fusion, att = model(seq_inputs, esm_embeddings, esm_mask) 

        
            loss = cls_loss_func(cls_probs, cls_labels)

        scaler.scale(loss).backward()

      
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        scaler.step(optimizer)
        scaler.update()

        batch_size = seq_inputs.size(0)
        total_loss_samples += loss.item() * batch_size   
        total_samples += batch_size
        all_cls_true.append(cls_labels.detach().cpu().numpy())
        all_cls_pred_prob.append(cls_probs.detach().cpu().numpy())

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss_samples / total_samples
    all_cls_true = np.concatenate(all_cls_true, axis=0)
    all_cls_pred_prob = np.concatenate(all_cls_pred_prob, axis=0)


    metrics_dict = performance(all_cls_true, all_cls_pred_prob)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return avg_loss, metrics_dict



def valid_one_epoch(model, dataloader, device, cls_loss_func, use_amp=False):


    model.eval() 
    total_loss_samples = 0.0
    total_samples = 0

    all_cls_true = []
    all_cls_hard_true = []
    all_cls_pred_prob = []



  
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validating")

        for batch in pbar:
            seq_inputs, reg_labels, cls_labels, esm_embeddings, esm_mask, weight, ids = batch
            seq_inputs = seq_inputs.to(device)
            cls_labels = cls_labels.to(device)
            weight = weight.to(device).squeeze()
            if esm_embeddings is not None:
                esm_embeddings = esm_embeddings.to(device)
            if esm_mask is not None:
                esm_mask = esm_mask.to(device)

            with torch.cuda.amp.autocast(enabled=use_amp):
               
                cls_probs, fusion, att = model(seq_inputs, esm_embeddings, esm_mask)  

              
                loss = cls_loss_func(cls_probs, cls_labels)  


            batch_size = seq_inputs.size(0)
            total_loss_samples += loss.item() * batch_size
            total_samples += batch_size

          
            all_cls_true.append(cls_labels.detach().cpu().numpy())
            all_cls_pred_prob.append(cls_probs.detach().cpu().numpy())

            pbar.set_postfix({"val_loss": f"{loss.item():.4f}"})

 
    avg_loss = total_loss_samples / total_samples

   
    all_cls_true = np.concatenate(all_cls_true, axis=0)
    all_cls_pred_prob = np.concatenate(all_cls_pred_prob, axis=0)


   
    metrics_dict = performance(all_cls_true, all_cls_pred_prob)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return avg_loss, metrics_dict




def parse_args():
    parser = argparse.ArgumentParser(description='Train Enzyme Classification Model')
    
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    default_csv = os.path.join(current_dir, '../../data/Tm/Tm.csv')
    default_esm = os.path.join(current_dir, '../../data/Tm/esm_embedding')
    default_save = os.path.join(current_dir, '../../train_model/Tm/savetrain')
    
   
    parser.add_argument('--csv_path', default=default_csv, type=str)
    parser.add_argument('--esm_path', default=default_esm, type=str)
    parser.add_argument('--save_dir', default=default_save, type=str)
    
    
    parser.add_argument('--seq_max_len', default=1024, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--espatience', default=18, type=int)
    parser.add_argument('--lr', default=2e-5, type=float)
    parser.add_argument('--weight_decay', default=1e-5, type=float)

    return parser.parse_args()



def main():
   

    seed_everything(42)
    args = parse_args()
    esm_features = load_embeddings(args.esm_path, max_length=args.seq_max_len)
    train_loader = cls_data_load_seq(
        csv_path=args.csv_path, batch_size=args.batch_size, split='Training', esm_loader=esm_features, seq_max_len=args.seq_max_len
    )
    valid_loader = cls_data_load_seq(
        csv_path=args.csv_path, batch_size=args.batch_size * 2, split='Validation', esm_loader=esm_features, seq_max_len=args.seq_max_len
    ) 
    cls_loss_func = nn.BCELoss()


    model = Mymodel_tm(vocab_size=vocab_size).to(device)


   
    def optimizer_fn(model):
        models_dict = {"main": model}
        optimizer, _ = build_optimizer_scheduler(models_dict,
            base_lr=args.lr, 
            weight_decay=args.weight_decay, 
            scheduler_type="plateau",
            mode="max", 
            factor=0.5, 
            patience=10, 
            verbose=True
        )
        return optimizer

   
    def scheduler_fn(optimizer):
        models_dict = {"main": model}
        _, scheduler = build_optimizer_scheduler(models_dict,
            base_lr=args.lr, 
            weight_decay=args.weight_decay, 
            scheduler_type="plateau",
            mode="max", 
            factor=0.5, 
            patience=10, 
            verbose=True
        )
        return scheduler

    
    model, optimizer, scheduler, progress, paths = prepare_training_independent(
        args.save_dir, model_name="tmcls", build_model_fn=lambda: model,
        optimizer_fn=optimizer_fn, scheduler_fn=scheduler_fn,
        restart=False, device=device
    )


    # --- EarlyStopping ---
    early_stopper = EarlyStopping(
        espatience=args.espatience, verbose=True, path=paths['best_model'],
        monitor="Accuracy", mode="max")
    historical_best_score = progress.get('best_val_accuracy', None)
    if historical_best_score is not None:
     
        early_stopper.best_score = historical_best_score
        if historical_best_score != -np.inf: 
            print(f"🔄 EarlyStopping best_score restored from log: {early_stopper.best_score:.4f}")

   
    start_epoch = progress['epoch']
    best_val_accuracy = progress['best_val_accuracy']

    print(f"🚀 Start Training from Epoch {start_epoch} / Max Epochs = {args.epochs}")

    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()

       
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, optimizer, device, cls_loss_func, 
            epoch_idx=epoch+1, total_epochs=args.epochs)

       
        valid_loss, valid_metrics = valid_one_epoch(
            model, valid_loader, device, cls_loss_func)

        
        if scheduler is not None:
            scheduler.step(valid_metrics['Accuracy'])

        
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

    print(f"\n🎯 Training finished. Best validation Accuracy = {best_val_accuracy:.4f}")
    print(f"📦 Best model saved to: {paths['best_model']}")

if __name__ == "__main__":
    main()




