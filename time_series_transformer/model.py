import torch 
import torch.nn as nn
import pytorch_lightning as pl

class TsTransformer(pl.LightningModule):
    
    def __init__(
        self,
        n_encoder_inputs,
        n_decoder_inputs,
        channels=512,
        dropout=0.1,
        lr=1e-3
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.lr = lr
        self.dropout = dropout