import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
import torch
import torch.nn as nn

from model_cls.modelcls import Mymodel_topt
from model_reg2040.modelreg import Mymodel_toptreg1
from model_regrest.modelreg import Mymodel_toptreg2





DEFAULT_CLS_CKPT = os.path.join(current_dir, '../best_model/Topt/classifier/model_cls.pt')
DEFAULT_REG1_CKPT = os.path.join(current_dir, '../best_model/Topt/Regressor_1/model_reg2040.pt')
DEFAULT_REG2_CKPT = os.path.join(current_dir, '../best_model/Topt/Regressor_2/model_reg120.pt')


# DEFAULT_CLS_CKPT = os.path.join(current_dir, '/home/dqw_gq/EnzOracle/train_model/Topt/savetrain/toptcls/model_epoch_20.pt')
# DEFAULT_REG1_CKPT = os.path.join(current_dir, '/home/dqw_gq/EnzOracle/train_model/Topt/savetrain/toptreg2040/model_epoch_30.pt')
# DEFAULT_REG2_CKPT = os.path.join(current_dir, '/home/dqw_gq/EnzOracle/train_model/Topt/savetrain/toptreg120/model_epoch_62.pt')


class FinalModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        cls_ckpt_path=DEFAULT_CLS_CKPT,
        reg1_ckpt_path=DEFAULT_REG1_CKPT, 
        reg2_ckpt_path=DEFAULT_REG2_CKPT,
        device='cpu'
    ):
        
        super().__init__()
        self.cls_model = Mymodel_topt(vocab_size=vocab_size).to(device)
        self.reg_model1 = Mymodel_toptreg1(vocab_size=vocab_size).to(device)
        self.reg_model2 = Mymodel_toptreg2(vocab_size=vocab_size).to(device)


        if cls_ckpt_path and os.path.exists(cls_ckpt_path):
            ckpt = torch.load(cls_ckpt_path, map_location=device)
            self.cls_model.load_state_dict(ckpt.get("model_state_dict", ckpt))
            print(f"✅ Loaded Classification model weights from: {cls_ckpt_path}")
        else:
            print(f"⚠️ Warning: Classification weights not found at {cls_ckpt_path}")

        if reg1_ckpt_path and os.path.exists(reg1_ckpt_path):
            ckpt = torch.load(reg1_ckpt_path, map_location=device)
            self.reg_model1.load_state_dict(ckpt.get("model_state_dict", ckpt))
            print(f"✅ Loaded Reg5060 model weights from: {reg1_ckpt_path}")
        else:
            print(f"⚠️ Warning: Reg5060 weights not found at {reg1_ckpt_path}")

        if reg2_ckpt_path and os.path.exists(reg2_ckpt_path):
            ckpt = torch.load(reg2_ckpt_path, map_location=device)
            self.reg_model2.load_state_dict(ckpt.get("model_state_dict", ckpt))
            print(f"✅ Loaded RegRest model weights from: {reg2_ckpt_path}")
        else:
            print(f"⚠️ Warning: RegRest weights not found at {reg2_ckpt_path}")

    def forward(self, seq_inputs, esm_embeddings, esm_mask):

        
        cls_prob, _, _ = self.cls_model(seq_inputs, esm_embeddings, esm_mask)  
        cls_prob = torch.sigmoid(cls_prob)

        
        y1, _, _ = self.reg_model1(seq_inputs, esm_embeddings, esm_mask)
        y1 = y1 * 20.0 + 20.0

        
        y2, _, _ = self.reg_model2(seq_inputs, esm_embeddings, esm_mask)
        y2 = y2 * 120.0 

        
        cls_prob = cls_prob.view(-1, 1)
        y1 = y1.view(-1, 1)
        y2 = y2.view(-1, 1)

        
        final_pred = cls_prob * y1 + (1 - cls_prob) * y2

        return final_pred, cls_prob, y1, y2