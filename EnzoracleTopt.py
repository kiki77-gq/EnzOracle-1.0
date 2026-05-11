import warnings
warnings.filterwarnings('ignore')
import os
import argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import esm

from Topt.modelTopt import FinalModel



def parse_args():
    parser = argparse.ArgumentParser(description="Topt Prediction Tool for User Sequences")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    
    parser.add_argument('--input', type=str, required=True, 
                        help="Path to input file (.txt, .csv, or .fasta)")
    parser.add_argument('--output', type=str, default="prediction_results.csv", 
                        help="Path to save the output CSV")
    
    
    parser.add_argument('--db_path', type=str, 
                        default=os.path.join(current_dir, 'data/Topt/Topt.csv'), 
                        help="Path to the training database for exact matching")
    
    
    parser.add_argument('--vocab_path', type=str, 
                        default=os.path.join(current_dir, 'tools/data_dict.npy'), 
                        help="Path to the sequence vocabulary dictionary")
    
    
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size for ESM and Model prediction")
    parser.add_argument('--seq_max_len', type=int, default=1024, help="Max sequence length")
    
    return parser.parse_args()


def parse_input_file(filepath):
    
    ext = os.path.splitext(filepath)[-1].lower()
    data = []
    
    if ext == '.fasta' or ext == '.fa':
        with open(filepath, 'r') as f:
            seq_id, seq = "", ""
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    if seq_id:
                        data.append({"ID": seq_id, "sequence": seq})
                    seq_id = line[1:]
                    seq = ""
                else:
                    seq += line
            if seq_id:
                data.append({"ID": seq_id, "sequence": seq})
                
    elif ext == '.csv':
        df = pd.read_csv(filepath)
        
        seq_col = 'sequence' if 'sequence' in df.columns else 'Sequence'
        id_col = 'ID' if 'ID' in df.columns else df.columns[0]
        for _, row in df.iterrows():
            data.append({"ID": str(row[id_col]), "sequence": str(row[seq_col])})
            
    elif ext == '.txt':
        with open(filepath, 'r') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if line:
                    data.append({"ID": f"Seq_{i+1}", "sequence": line})
    else:
        raise ValueError("Unsupported file format! Please use .txt, .csv, or .fasta")
        
    return data


def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Initializing Prediction Pipeline on {device}...")

    
    input_data = parse_input_file(args.input)
    print(f"📥 Loaded {len(input_data)} sequences from user input.")

    
    db_dict = {}
    if os.path.exists(args.db_path):
        db_df = pd.read_csv(args.db_path)
        seq_col = 'sequence' if 'sequence' in db_df.columns else 'Sequence'
        
        db_dict = dict(zip(db_df[seq_col], db_df['topt']))
        print(f"📚 Loaded database with {len(db_dict)} sequences for exact matching.")
    else:
        print("⚠️ Database not found. Skipping exact match checking.")

    results = []
    to_predict = []

    
    for item in input_data:
        seq = item['sequence']
        if seq in db_dict:
            results.append({
                "ID": item['ID'], 
                "Sequence": seq, 
                "Topt": db_dict[seq], 
                "Source": "Database Exact Match"
            })
        else:
            to_predict.append(item)

    print(f"✅ Found {len(results)} exact matches in database. {len(to_predict)} sequences need prediction.")

    
    if len(to_predict) > 0:
        print("🧠 Loading ESM-2 Model for feature extraction...")
        esm_model, alphabet = esm.pretrained.load_model_and_alphabet('esm2_t33_650M_UR50D')
        esm_model = esm_model.to(device).eval()
        batch_converter = alphabet.get_batch_converter()

        print("🔮 Loading Final Tm Prediction Model...")
        vocab = np.load(args.vocab_path, allow_pickle=True).item()
        vocab_size = len(vocab)
        final_model = FinalModel(vocab_size=vocab_size, device=device)
        final_model = final_model.to(device).eval()

        
        print("⚙️ Running Predictions...")
        for i in tqdm(range(0, len(to_predict), args.batch_size)):
            batch = to_predict[i : i + args.batch_size]
            
            
            esm_data = [(item['ID'], item['sequence']) for item in batch]
            _, _, batch_tokens = batch_converter(esm_data)
            batch_tokens = batch_tokens.to(device)

            with torch.no_grad():
                esm_results = esm_model(batch_tokens, repr_layers=[33], return_contacts=False)
                
                tokens_embs = esm_results['representations'][33]

            
            batch_seq_inputs = []
            batch_esm_feats = []
            batch_esm_masks = []

            for b_idx, item in enumerate(batch):
                seq = item['sequence']
                
                
                trunc_seq = seq[:args.seq_max_len - 2]
                trunc_seq = '-' + trunc_seq + '-'
                padded_seq = trunc_seq.ljust(args.seq_max_len, '-')
                seq_input = [vocab.get(s, vocab.get('-')) for s in padded_seq] 
                batch_seq_inputs.append(seq_input)

                
                feat = tokens_embs[b_idx].cpu().numpy()
                padded_feat = np.zeros((args.seq_max_len, 1280))
                if feat.shape[0] > args.seq_max_len:
                    padded_feat = feat[:args.seq_max_len, :]
                else:
                    padded_feat[:feat.shape[0], :] = feat
                batch_esm_feats.append(padded_feat)
                
                
                mask = (np.abs(padded_feat).sum(axis=1) != 0)
                batch_esm_masks.append(mask)

            
            t_seq_inputs = torch.LongTensor(batch_seq_inputs).to(device)
            t_esm_feats = torch.FloatTensor(np.array(batch_esm_feats)).to(device)
            t_esm_masks = torch.BoolTensor(np.array(batch_esm_masks)).to(device)

            
            with torch.no_grad():
                final_pred, _, _, _ = final_model(t_seq_inputs, t_esm_feats, t_esm_masks)
                final_pred = final_pred.view(-1).cpu().numpy()

            
            for item, pred_val in zip(batch, final_pred):
                results.append({
                    "ID": item['ID'],
                    "Sequence": item['sequence'],
                    "Topt": round(float(pred_val), 4),
                    "Source": "AI Model Prediction"
                })

            
            torch.cuda.empty_cache()

    
    out_df = pd.DataFrame(results)
    out_df.to_csv(args.output, index=False)
    print(f"🎉 All done! Results seamlessly saved to: {args.output}")

if __name__ == "__main__":
    main()