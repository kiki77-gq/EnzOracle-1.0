import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer



use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")



class PositionalEncoding_padding(nn.Module):
    def __init__(self, d_model=128, max_len=1024, dropout=0.2):
        super(PositionalEncoding_padding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pad = torch.zeros(1024,128)
        pad[:pe.shape[0], :] = pe

        pe = pad.unsqueeze(0).transpose(0, 1).to(device)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x.to(device) + self.pe[:x.size(0), :].to(device)
        return self.dropout(x)




def get_attn_pad_mask(seq_q,seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # batch_size, 1, len_k   False is masked
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # batch_size, len_q, len_k


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(64)
        scores.masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn



class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.use_cuda = use_cuda
        self.W_Q = nn.Linear(128, 64 * 4, bias=False)
        self.W_K = nn.Linear(128, 64 * 4, bias=False)
        self.W_V = nn.Linear(128, 64 * 4, bias=False)
        self.fc = nn.Linear(4 * 64, 128, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask):

        residual, batch_size = input_Q, input_Q.size(0)
        Q = self.W_Q(input_Q).view(batch_size, -1, 4, 64).transpose(1, 2)
        K = self.W_K(input_K).view(batch_size, -1, 4, 64).transpose(1, 2)
        V = self.W_V(input_V).view(batch_size, -1, 4, 64).transpose(1, 2)

        attn_mask = attn_mask.unsqueeze(1).repeat(1, 4, 1, 1)

        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1, 4 * 64)
        output = self.fc(context)
        return nn.LayerNorm(128).to(device)(output + residual), attn



class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.use_cuda = use_cuda
        self.fc = nn.Sequential(
            nn.Linear(128, 512, bias=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 128, bias=False)
        )

    def forward(self, inputs):
        residual = inputs
        output = self.fc(inputs)
        output = nn.Dropout(0.1)(output)
        return nn.LayerNorm(128).to(device)(output + residual)  # [batch_size, seq_len, d_model]



class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()
        self.dropout = nn.Dropout(0.2)
    def forward(self, enc_inputs, enc_self_attn_mask):

        # enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
        
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,
                                               enc_self_attn_mask)  # enc_inputs to same Q,K,V
        enc_outputs1 = enc_inputs+self.dropout(enc_outputs)
        enc_outputs1 = nn.LayerNorm(128).to(device)(enc_outputs1)
        enc_outputs = self.pos_ffn(enc_outputs1)  # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs, attn



class Encoder_padding(nn.Module):
    def __init__(self, vocab_size, pad_idx=0):
        self.pad_idx = pad_idx
        super(Encoder_padding, self).__init__()
        self.src_emb = nn.Embedding(vocab_size, 128)
        self.pos_emb_padding = PositionalEncoding_padding(128,max_len=1024)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(3)])

    def forward(self, enc_inputs):
        enc_outputs = self.src_emb(enc_inputs)
        batch_size, seq_len = enc_inputs.size()
        enc_pad = torch.zeros(batch_size, 1024, 128, device=enc_inputs.device)
        enc_pad[:, :seq_len, :] = enc_outputs
        enc_outputs = enc_pad
        enc_outputs = self.pos_emb_padding(enc_outputs.transpose(0, 1)).transpose(0, 1)
        # create mask: True for valid tokens, False for padding
        mask = (enc_inputs != self.pad_idx)
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        
        enc_self_attns = []
        for layer in self.layers:
            # enc_outputs: batch_size, src_len, d_model, enc_self_attn: batch_size, n_heads, src_len, src_len
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns, mask

    

class AttentionBlock(nn.Module):
    def __init__(self, hid_dim, num_heads, dropout):
        super().__init__()
        assert hid_dim % num_heads == 0

        self.hid_dim = hid_dim
        self.num_heads = num_heads
        self.head_dim = hid_dim // num_heads
        self.scale = (self.head_dim) ** -0.5  

        self.f_q = nn.Linear(hid_dim, hid_dim)
        self.f_k = nn.Linear(hid_dim, hid_dim)
        self.f_v = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(hid_dim, hid_dim)

    def forward(self, query, key, value, mask=None, return_attn=False):
        B, L, _ = query.shape

        q = self.f_q(query).view(B, L, self.num_heads, self.head_dim)
        k = self.f_k(key).view(B, L, self.num_heads, self.head_dim)
        v = self.f_v(value).view(B, L, self.num_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        weighted = torch.matmul(attn, v)
        weighted = weighted.transpose(1, 2).contiguous()
        weighted = weighted.reshape(B, L, self.hid_dim)

        output = self.fc_out(weighted)

        if return_attn:
            return output, attn
        return output



class CrossAttentionBlock(nn.Module):
    def __init__(self, hid_dim, dropout, num_heads=2):
        super(CrossAttentionBlock, self).__init__()
        self.att = AttentionBlock(hid_dim=hid_dim, num_heads=num_heads, dropout=dropout)
        
        self.linear_seq = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
        )

        self.linear_esm = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
        )

        self.norm_seq = nn.LayerNorm(hid_dim)
        self.norm_esm = nn.LayerNorm(hid_dim)

    def forward(self, seq_features, esm_features, esm_mask):
       

        if self.training:
            seq_att = self.att(seq_features, esm_features, esm_features, mask=esm_mask)
            esm_att = self.att(esm_features, seq_features, seq_features, mask=esm_mask)
            att_seq_esm = att_esm_seq = None  
        else:
            seq_att, att_seq_esm = self.att(seq_features, esm_features, esm_features, mask=esm_mask, return_attn=True)
            esm_att, att_esm_seq = self.att(esm_features, seq_features, seq_features, mask=esm_mask, return_attn=True)

        # Linear + residual + norm
        seq_features = self.norm_seq(self.linear_seq(seq_att) + seq_features)
        esm_features = self.norm_esm(self.linear_esm(esm_att) + esm_features)

        return seq_features, esm_features, att_seq_esm, att_esm_seq

        


class ExtraEncoder(nn.Module):
    """Stacked TransformerEncoder layers (PyTorch) with batch_first=True"""
    def __init__(self, d_model, num_layers=2, nhead=2, dim_feedforward=512, dropout=0.1):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x, src_key_padding_mask=None):
        # src_key_padding_mask: True for padded positions
        return self.encoder(x, src_key_padding_mask=src_key_padding_mask)


class AttentionPooling(nn.Module):
    """Attention pooling -> weighted sum over tokens"""
    def __init__(self, d_model):
        super().__init__()
        self.att = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1)
        )

    def forward(self, x, mask=None):
        # x: [B, L, D]
        score = self.att(x).squeeze(-1)  # [B, L]
        if mask is not None:
            # mask: True for valid tokens -> mask out invalid
            score = score.masked_fill(~mask, -1e9)
        weights = torch.softmax(score, dim=-1).unsqueeze(-1)  # [B, L, 1]
        pooled = torch.sum(weights * x, dim=1)  # [B, D]
        return pooled, weights.squeeze(-1)  # returning weights optional




class predictModel(nn.Module):
    def __init__(self, seq_len, enc_dim=128, esm_dim=1280, proj_dim=512, conv_out=512, num_extra_encoder_layers=2):
       

        super().__init__()
        self.seq_len = seq_len
        self.enc_dim = enc_dim
        self.esm_dim = esm_dim
        self.proj_dim = proj_dim
        self.conv_out = conv_out


        
        self.seq_proj = nn.Linear(enc_dim, proj_dim)
        self.esm_proj = nn.Linear(esm_dim, proj_dim)

        # Cross attention + Extra encoder
        self.cross_attn = CrossAttentionBlock(hid_dim=proj_dim, num_heads=2, dropout=0.1)
        self.extra_encoder = ExtraEncoder(d_model=proj_dim, num_layers=num_extra_encoder_layers, nhead=2,
                                          dim_feedforward=proj_dim * 4, dropout=0.1)

        # Attention pooling
        self.att_pool = AttentionPooling(proj_dim)

        # Conv branch (operate on concat(seq, esm) in original input_dim space)
        self.conv = nn.Sequential(
            nn.Conv1d(enc_dim + esm_dim, (enc_dim + esm_dim) * 2, 3, stride=1, padding=1),
            nn.LayerNorm(((enc_dim + esm_dim) * 2, seq_len)),
            nn.LeakyReLU(True),

            nn.Conv1d((enc_dim + esm_dim) * 2, (enc_dim + esm_dim) * 2, 3, stride=1, padding=1),
            nn.LayerNorm(((enc_dim + esm_dim) * 2, seq_len)),
            nn.LeakyReLU(True),

            nn.Conv1d((enc_dim + esm_dim) * 2, conv_out, 3, stride=1, padding=1),
            nn.LayerNorm((conv_out, seq_len)),
            nn.LeakyReLU(True),
        )

        # fusion dimension: proj_dim (att branch) + conv_out
        self.cls_dim = proj_dim + conv_out


        
        self.regressor = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(self.cls_dim, max(self.cls_dim // 4, 64)),
            nn.LeakyReLU(True),
            nn.Dropout(p=0.3),
            nn.Linear(max(self.cls_dim // 4, 64), max(self.cls_dim // 4, 64)),
            nn.LeakyReLU(True),
            nn.Dropout(p=0.3),
            nn.Linear(max(self.cls_dim // 4, 64), 1),  
        )

        self._initialize_weights()

    def forward(self, seq_feats, esm_feats=None, esm_mask=None):
        

        B, L, _ = seq_feats.size()
        mask_used = esm_mask  

        if esm_feats is not None:
            esm_feats = esm_feats.to(seq_feats.device).float()
            
            seq_p = self.seq_proj(seq_feats)  # [B, L, proj_dim]
            esm_p = self.esm_proj(esm_feats)  # [B, L, proj_dim]

            # Cross-attention: seq queries, esm key/value
            seq_att, esm_att, _, _ = self.cross_attn(seq_p, esm_p, mask_used)  # [B, L, proj_dim]

            # Extra transformer encoder
            src_key_padding_mask = ~mask_used if mask_used is not None else None
            seq_enc = self.extra_encoder(seq_att, src_key_padding_mask=src_key_padding_mask)  # [B, L, proj_dim]

            # Attention pooling on seq_enc
            pooled_att, att_weights = self.att_pool(seq_enc, mask=mask_used)  # [B, proj_dim]

            # Conv branch on raw concat features (seq + esm)
            concat_feats = torch.cat([seq_feats, esm_feats], dim=2)  # [B, L, enc+esm]
            conv_in = concat_feats.permute(0, 2, 1)  # [B, C, L] for Conv1d
            conv_out = self.conv(conv_in).permute(0, 2, 1)  # [B, L, conv_out]
            if mask_used is not None:
                conv_out = conv_out * mask_used.unsqueeze(2).float()
            conv_pooled = torch.max(conv_out, dim=1)[0]  # [B, conv_out]

            
            fusion = torch.cat([pooled_att, conv_pooled], dim=1)  # [B, proj_dim+conv_out]

        else:
            
            seq_p = self.seq_proj(seq_feats)
            src_key_padding_mask = ~mask_used if mask_used is not None else None
            seq_enc = self.extra_encoder(seq_p, src_key_padding_mask=src_key_padding_mask)
            pooled_att, att_weights = self.att_pool(seq_enc, mask=mask_used)

            
            conv_in = seq_feats.permute(0, 2, 1)
            pad = torch.zeros(B, self.esm_dim, L, device=seq_feats.device)
            conv_in_cat = torch.cat([conv_in, pad], dim=1)  # [B, enc+esm, L]
            conv_out = self.conv(conv_in_cat).permute(0, 2, 1)
            if mask_used is not None:
                conv_out = conv_out * mask_used.unsqueeze(2).float()
            conv_pooled = torch.max(conv_out, dim=1)[0]

            fusion = torch.cat([pooled_att, conv_pooled], dim=1)

        pred_reg = self.regressor(fusion).squeeze(1)

        
        return pred_reg, fusion, att_weights


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm1d):
                continue
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)



class Mymodel_toptreg1(nn.Module):
    def __init__(self, vocab_size):
        super(Mymodel_toptreg1, self).__init__()
        self.use_cuda = use_cuda

        self.encoder = Encoder_padding(vocab_size=vocab_size).to(device)
        self.predictor = predictModel(seq_len=1024,
                                      enc_dim=128,
                                      esm_dim=1280,
                                      proj_dim=512,
                                      conv_out=512,
                                      num_extra_encoder_layers=3).to(device)
        

    def forward(self, seq_inputs, esm_embeddings=None, esm_mask=None):

        enc_outputs, enc_attn, _= self.encoder(seq_inputs)
        if esm_embeddings is not None and esm_mask is not None:
            # ensure esm_embeddings is float tensor on same device
            if not isinstance(esm_embeddings, torch.Tensor):
                esm_embeddings = esm_embeddings.to(device).float()
            else:
                esm_embeddings = esm_embeddings.to(device).float()
            esm_mask = esm_mask.to(device)
            # pass encoder outputs and esm separately
            pred_reg, fusion, att_weights = self.predictor(enc_outputs, esm_embeddings, esm_mask=esm_mask)
        else:
            pred_reg, fusion, att_weights = self.predictor(enc_outputs, None, mask=(seq_inputs != 0), esm_mask=None)
        return pred_reg, fusion, att_weights










        











