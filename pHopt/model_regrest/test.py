import warnings
warnings.filterwarnings('ignore')
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32,garbage_collection_threshold:0.6"
import random
import numpy as np
import pandas as pd
import torch
from modelreg import *
from tqdm import tqdm
import time
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score)
from scipy.stats import pearsonr, spearmanr
import argparse
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

class WeightedMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, weight=None):
       
        
        loss = (pred - target) ** 2
        
        if weight is not None:
            
            weight = weight.to(pred.device).view_as(loss)
            loss = loss * weight
            
        return loss.mean()
    

def write_logfile(test_loss, test_metrics, logfile):
   

    columns = [
        "test_loss", 
        "test_r2", "test_pearson", "test_spearman", "test_rmse", "test_mae", "test_ci"
    ]

    values = [
        test_loss,
        test_metrics.get("R2", np.nan),
        test_metrics.get("Pearson", np.nan),
        test_metrics.get("Spearman", np.nan),
        test_metrics.get("RMSE", np.nan),
        test_metrics.get("MAE", np.nan),
        test_metrics.get("CI", np.nan)
    ]

    df = pd.DataFrame([values], columns=columns)

    if not os.path.exists(logfile):
        df.to_csv(logfile, index=False, float_format="%.4f")
    else:
        df.to_csv(logfile, mode='a', header=False, index=False, float_format="%.4f")


def log_epoch_results(start_time, test_loss, test_metrics):
    

    elapsed = time.time() - start_time
    print(
        f"Test Loss: {test_loss:.4f}, R2: {test_metrics.get('R2',0):.4f}, MAE: {test_metrics.get('MAE',0):.4f} | "
        f"Time: {elapsed:.2f}s"
    )




def test_one_epoch(model, dataloader, device, reg_loss_func, use_amp=False):
  

    model.eval()  
    total_loss_samples = 0.0
    total_samples = 0

    all_reg_true = []
    all_reg_pred = []
    all_ids = []


    
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

                
                reg_loss = reg_loss_func(pred_reg, reg_labels, weight=weight)

                
                loss = reg_loss

            batch_size = seq_inputs.size(0)
            total_loss_samples += loss.item() * batch_size
            total_samples += batch_size

             

            all_ids.extend(ids)
            all_reg_true.append(reg_labels.detach().cpu().numpy())
            all_reg_pred.append(pred_reg.detach().cpu().numpy())

            pbar.set_postfix({"val_loss": f"{loss.item():.4f}"})

    
    avg_loss = total_loss_samples / total_samples

   
    all_reg_true = np.concatenate(all_reg_true, axis=0)
    all_reg_pred = np.concatenate(all_reg_pred, axis=0)


    

    all_reg_true = all_reg_true * 11.0 + 1.0
    all_reg_pred = all_reg_pred * 11.0 + 1.0

    

    metrics_dict = performance(all_reg_true, all_reg_pred)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return avg_loss, metrics_dict, all_reg_true, all_reg_pred, all_ids



def parse_args():
    parser = argparse.ArgumentParser(description='Test Enzyme Regression Model')
    

    current_dir = os.path.dirname(os.path.abspath(__file__))
    

    default_csv = os.path.join(current_dir, '../../data/pHopt/pHopt_rest.csv')
    default_esm = os.path.join(current_dir, '../../data/pHopt/esm_embedding')
  
    default_output_dir = os.path.join(current_dir, '../../train_model/pHopt/savetrain/phreg12')
    default_model_path = os.path.join(default_output_dir, 'model_best.pt')
    
    parser.add_argument('--csv_path', default=default_csv, type=str, help="Path to the dataset CSV file")
    parser.add_argument('--esm_path', default=default_esm, type=str, help="Directory containing ESM feature embeddings")
    parser.add_argument('--model_path', default=default_model_path, type=str, help="Path to the best model checkpoint")
    parser.add_argument('--output_dir', default=default_output_dir, type=str, help="Directory to save test results and logs")
    
    parser.add_argument('--seq_max_len', default=1024, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    
    return parser.parse_args()




def main():


    seed_everything(42)
    args = parse_args()
    esm_features = load_embeddings(args.esm_path, max_length=args.seq_max_len)
    test_loader = data_load_seq(csv_path=args.csv_path, split='Testing', batch_size=args.batch_size, esm_loader=esm_features, seq_max_len=args.seq_max_len)
    reg_loss_func = WeightedMSELoss()

    model = Mymodel_phreg2(vocab_size=vocab_size).to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    print("✅ Model weights loaded successfully.")
    log_path = os.path.join(args.output_dir, "test_0107.csv")


    start_time = time.time()


    
    test_loss, test_metrics, reg_true, reg_pred, ids = test_one_epoch(
        model, test_loader, device, reg_loss_func)
    

    
    save_df = pd.DataFrame({
        "ID": ids,
        "pHopt": reg_true.flatten(),
        "prediction": reg_pred.flatten()
    })
    pred_csv_path = os.path.join(args.output_dir, "test_pred_results0107.csv")
    save_df.to_csv(pred_csv_path, index=False)
    print("✅ Saved prediction results to CSV.")



    
    write_logfile(test_loss, test_metrics, log_path)
    log_epoch_results(start_time, test_loss, test_metrics)



if __name__ == "__main__":
    main()







