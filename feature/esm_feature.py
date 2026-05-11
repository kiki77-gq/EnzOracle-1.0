import torch
import warnings
warnings.filterwarnings('ignore')
import esm
import os
import pandas as pd  
import numpy as np 
from tqdm import tqdm
import os
current_dir = os.path.dirname(os.path.abspath(__file__))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


model_name = 'esm2_t33_650M_UR50D'
model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
model = model.to(device)  
model.eval()


csv_path = os.path.join(current_dir, '../data/Topt/Topt.csv')
#../data/Tm/Tm.csv
#
#../data/pHopt/pHopt.csv
output_directory = os.path.join(current_dir, '../data/Topt/esm_embedding')
#../data/Tm/esm_embedding
#
#../data/pHopt/esm_embedding
os.makedirs(output_directory, exist_ok=True)


df = pd.read_csv(csv_path, header=0)
df.rename(columns={df.columns[0]: "ID"}, inplace=True)
print("test:", df.columns.tolist())
batch_size = 200
sequences = list(zip(df['ID'], df['sequence']))
for i in tqdm(range(0, len(sequences), batch_size)):
    batch = sequences[i:i+batch_size]
    
    for protein_id, sequence in batch:


        data = [(protein_id, sequence)]
        batch_converter = alphabet.get_batch_converter()
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to(device) 

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=False)
        
        tokens_embs = results['representations'][33].cpu().numpy()  


        np.save(os.path.join(output_directory, f'{protein_id}.npy'), tokens_embs)
        print(f'Saved embeddings for {protein_id}')

    torch.cuda.empty_cache()
