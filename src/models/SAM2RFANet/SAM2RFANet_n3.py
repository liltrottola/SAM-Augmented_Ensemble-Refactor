import torch
import torch.nn as nn
from SAM2RFANet import SAM2RFANet

class ChannelReduction(nn.Module):
    """
    Riduce un tensore di input con N canali (RGB + Priors) ai 3 canali 
    richiesti dal backbone di SAM2 (RGB) prima dell'estrazione feature.
    """
    def __init__(self, in_channels, kernel_size=1):
        super(ChannelReduction, self).__init__()
        # Il padding deve essere kernel_size // 2 per mantenere intatta H, W
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, 3, kernel_size=kernel_size, padding=padding)
        # Inizializziamo a zero l'ultimo layer in modo che, se usiamo residual, 
        # all'inizio l'output di questo layer sia nullo
        nn.init.constant_(self.conv.weight, 0)
        nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        return self.conv(x)

class SAM2RFANet_N3(nn.Module):
    def __init__(self, in_channels, residual=False, kernel_size=1, checkpoint_path=None):
        super(SAM2RFANet_N3, self).__init__()
        self.residual = residual
        
        # Blocco di riduzione canali 
        self.reduce = ChannelReduction(in_channels, kernel_size)
        
        # Modello Base
        self.sam2rfanet = SAM2RFANet(checkpoint_path=checkpoint_path)

    def forward(self, x):
        # 1. Estrai la sola componente RGB per il calcolo residual (primi 3 canali)
        rgb_base = x[:, :3, :, :]
        
        # 2. Passa tutto il volume (N canali) attraverso il ChannelReduction
        reduced = self.reduce(x)
        
        # 3. Fusion (Residual o Direct)
        if self.residual:
            out_fusion = rgb_base + reduced
        else:
            out_fusion = reduced
            
        # 4. Inferenza nel modello SAM2-RFANet
        return self.sam2rfanet(out_fusion)

