import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32,garbage_collection_threshold:0.6"
import random
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import time
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score)
from scipy.stats import pearsonr, spearmanr
from dataset import load_embeddings, data_load, vocab_size
from modelpHopt import FinalModel
import argparse



def parse_args():
    parser = argparse.ArgumentParser(description='Test Integrated Final Enzyme Model')
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    
    default_csv = os.path.join(current_dir, '../data/pHopt/pHopt.csv')
    default_esm = os.path.join(current_dir, '../data/pHopt/esm_embedding')
    default_output_dir = os.path.join(current_dir, '../data/pHopt')
    
    parser.add_argument('--csv_path', default=default_csv, type=str, help="Path to the dataset CSV file")
    parser.add_argument('--esm_path', default=default_esm, type=str, help="Directory containing ESM feature embeddings")
    parser.add_argument('--output_dir', default=default_output_dir, type=str, help="Directory to save final prediction results")
    
    parser.add_argument('--seq_max_len', default=1024, type=int, help="Maximum length for sequence padding")
    parser.add_argument('--batch_size', default=6, type=int, help="Batch size for testing")
    
    return parser.parse_args()


def seed_everything(seed=42):
   
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



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

 


def final_test_safe(final_model, test_loader, device):
  
    final_model.eval()

    all_ids = []
    all_final = []
    all_cls_prob = []
    all_y1 = []
    all_y2 = []
    all_true = [] 

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Final Testing")

        for batch in pbar:
            # unpack batch
            seq_inputs, reg_labels, cls_labels, esm_embeddings, esm_mask, weight, ids = batch

            seq_inputs = seq_inputs.to(device)
            if esm_embeddings is not None:
                esm_embeddings = esm_embeddings.to(device)
            if esm_mask is not None:
                esm_mask = esm_mask.to(device)

            
            final_pred, cls_prob, y1, y2 = final_model(seq_inputs, esm_embeddings, esm_mask)

       
            if isinstance(ids, torch.Tensor):
                ids = ids.view(-1).cpu().tolist()
            elif isinstance(ids, np.ndarray):
                ids = ids.flatten().tolist()

            
            final_pred = final_pred.view(-1).cpu().numpy()
            cls_prob   = cls_prob.view(-1).cpu().numpy()
            y1 = y1.view(-1).cpu().numpy()
            y2 = y2.view(-1).cpu().numpy()
            true_vals = reg_labels.view(-1).cpu().numpy()

            
            all_ids.extend(ids)
            all_final.extend(final_pred)
            all_cls_prob.extend(cls_prob)
            all_y1.extend(y1)
            all_y2.extend(y2)
            all_true.extend(true_vals)



    all_true = np.array(all_true)
    all_final = np.array(all_final)


    metrics_dict = performance(all_true, all_final)

    print(metrics_dict)

    
    df = pd.DataFrame({
        "ID": all_ids,
        "pHopt": all_true,
        "prediction": all_final,
        "cls_prob": all_cls_prob,
        "reg1_pred": all_y1,
        "reg2_pred": all_y2
    })

    return df, metrics_dict



# =========================
#        Main
# =========================
def main():
    seed_everything(42)
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("🚀 Preparing Data...")
    esm_features = load_embeddings(args.esm_path, max_length=args.seq_max_len)
    test_loader = data_load(
        csv_path=args.csv_path, 
        batch_size=args.batch_size, 
        split='Testing', 
        esm_loader=esm_features,
        seq_max_len=args.seq_max_len
    )


    print("📦 Initializing FinalTMModel and loading sub-model weights...")
    final_model = FinalModel(
        vocab_size=vocab_size,  
        device=device
    )


   
    # =======================
    #      Final Testing
    # =======================
    
    start_time = time.time()
    df, metrics = final_test_safe(final_model, test_loader, device)

    
    out_csv = os.path.join(args.output_dir, "pHopt_final_prediction.csv")
    df.to_csv(out_csv, index=False)
    print(f"✅ Final prediction saved to: {out_csv}")

    metrics_csv = os.path.join(args.output_dir, "final_metrics_log.csv")
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(metrics_csv, index=False)
    
    print(f"⏱️ Total testing time: {time.time() - start_time:.2f}s")



if __name__ == "__main__":
    main()
