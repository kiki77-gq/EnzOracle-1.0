import os
import numpy as np
import pandas as pd
import torch
import torch.utils.data as Data



current_dir = os.path.dirname(os.path.abspath(__file__))
vocab_path = os.path.join(current_dir, '../../tools/data_dict.npy')
vocab = np.load(vocab_path, allow_pickle=True).item()
vocab_size = len(vocab)



class load_embeddings:

    def __init__(self, folder_path, max_length):
        self.folder_path = folder_path
        self.max_length = max_length
        self.seq_ids = [os.path.splitext(f)[0] 
                        for f in os.listdir(folder_path) 
                        if f.endswith(('.npy', '.pt', '.tensor'))]

    def __len__(self):
        return len(self.seq_ids)

    def __getitem__(self, seq_id):
        
        
        file_path = None
        for ext in ('.npy', '.pt', '.tensor'):
            path = os.path.join(self.folder_path, seq_id + ext)
            if os.path.exists(path):
                file_path = path
                break
        if file_path is None:
            raise KeyError(f"No embedding found for {seq_id}")

        
        if file_path.endswith('.npy'):
            feature = np.load(file_path)
        else:
            feature = torch.load(file_path, map_location='cpu') 
            if isinstance(feature, torch.Tensor):
                feature = feature.cpu().numpy() 

        feature = np.array(feature)

        if feature.ndim == 3 and feature.shape[0] == 1:
            feature = feature[0]

        L, D = feature.shape

        
        if L > self.max_length:
            feature = feature[:self.max_length, :]
        elif L < self.max_length:
            pad = np.zeros((self.max_length - L, D))
            feature = np.vstack([feature, pad])


        
        x = torch.tensor(feature, dtype=torch.float32) # (1, dim, max_length)
        mask = torch.tensor((np.abs(feature).sum(axis=1) != 0), dtype=torch.bool)

        return x, mask


class SeqDataset(Data.Dataset):
    def __init__(self, data, vocab, seq_max_len, esm_loader=None):
       

        self.data = data
        self.vocab = vocab
        self.seq_max_len = seq_max_len
        self.esm_loader = esm_loader

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        seq = row['sequence'][:self.seq_max_len - 2]  
        seq = '-' + seq + '-'                         
        seq = seq.ljust(self.seq_max_len, '-')       
        seq_input = [self.vocab[n] for n in seq]     


        tm_min = 20.0
        tm_max = 40.0
        reg_label = (row['topt'] - tm_min) / (tm_max - tm_min)


        seq_id = str(row['ID'])  
        weight = float(row['weight'])

        if self.esm_loader is not None:
            seq_id = str(row['ID'])
            try:
                esm_feature, esm_mask = self.esm_loader[seq_id]
                
                if not isinstance(esm_feature, torch.Tensor):
                    esm_feature = torch.tensor(esm_feature, dtype=torch.float32)
                else:
                    esm_feature = esm_feature.clone().detach().float()
                
                if not isinstance(esm_mask, torch.Tensor):
                    esm_mask = torch.tensor(esm_mask, dtype=torch.bool)
                else:
                    esm_mask = esm_mask.clone().detach().bool()
            except KeyError:
                print(f"Warning: ESM feature not found for {seq_id}")
                esm_feature = torch.zeros(self.seq_max_len, 1280, dtype=torch.float32)
                esm_mask = torch.zeros(self.seq_max_len, dtype=torch.bool)
        else:
            esm_feature = torch.zeros(self.seq_max_len, 1280, dtype=torch.float32)
            esm_mask = torch.zeros(self.seq_max_len, dtype=torch.bool)


        
        return torch.LongTensor(seq_input), torch.FloatTensor([reg_label]), esm_feature, esm_mask, torch.FloatTensor([weight]), seq_id  



def data_load_seq(csv_path, batch_size, split='Training',esm_loader=None, seq_max_len=1024):


    data = pd.read_csv(csv_path)
    
    
    if split not in data['Split'].unique():
        raise ValueError(f"Specified split '{split}' not found in the CSV. Available splits: {data['Split'].unique()}")
    
    
    data_split = data[data['Split'] == split].reset_index()
    
    if len(data_split) == 0:
        raise ValueError(f"No samples found for split '{split}'. Please check the CSV file.")

    
    dataset = SeqDataset(data_split, vocab, seq_max_len=seq_max_len, esm_loader=esm_loader)
    
    loader = Data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split.lower() == 'training'),  
        num_workers=0,
        drop_last=False
    )
    
    print(f"DataLoader created for split='{split}', {len(dataset)} samples, batch_size={batch_size}")
    return loader