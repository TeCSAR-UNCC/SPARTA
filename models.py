import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000, device='cuda:0'):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)
        self.encoding = self.encoding.to(device)

    def forward(self, x):
        # Add positional encoding to the input tensor
        return x + self.encoding[:, :x.size(1)].detach()
    
class TransformerEncoder(nn.Module):
    def __init__(self, d_model, num_heads, d_feedforward, num_layers, max_len=1000, device='cuda:0'):
        super(TransformerEncoder, self).__init__()
        self.positional_encoding = PositionalEncoding(d_model, max_len, device)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, num_heads, d_feedforward) 
            for _ in range(num_layers)
        ])

    def forward(self, x):
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x)
        return x
    
class TransformerDecoder(nn.Module):
    def __init__(self, d_model, num_heads, d_feedforward, num_layers, max_len=1000, device='cuda:0'):
        super(TransformerDecoder, self).__init__()
        self.positional_encoding = PositionalEncoding(d_model, max_len, device)
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model, num_heads, d_feedforward) 
            for _ in range(num_layers)
        ])

    def forward(self, tgt, memory):
        tgt = self.positional_encoding(tgt)
        for layer in self.layers:
            tgt = layer(tgt, memory)
        return tgt
    
class SPARTA_F(nn.Module):
    def __init__(self, d_model, num_heads, d_feedforward, num_layers, max_len=1000, device='cuda:0'):
        super(SPARTA_F, self).__init__()
        self.encoder = TransformerEncoder(d_model, num_heads, d_feedforward, num_layers, max_len, device)
        self.decoder = TransformerDecoder(d_model, num_heads, d_feedforward, num_layers, max_len, device)

    def forward(self, src, tgt):
        memory = self.encoder.forward(src)
        output = self.decoder.forward(tgt, memory)
        return output
    
        
class SPARTA_C(nn.Module):
    def __init__(self, d_model, num_heads, d_feedforward, num_layers, max_len=1000, device='cuda:0', dropout=0.1):
        super(SPARTA_C, self).__init__()
        self.encoder_layers = nn.TransformerEncoderLayer(d_model, num_heads, d_feedforward, dropout=dropout, batch_first=True) 

        self.decoder_layers = nn.TransformerDecoderLayer(d_model, num_heads, d_feedforward, dropout=dropout, batch_first=True) 

        self.positional_encoding = PositionalEncoding(d_model, max_len, device)
        self.encoder = nn.TransformerEncoder(self.encoder_layers, num_layers=num_layers)
        self.decoder = nn.TransformerDecoder(self.decoder_layers, num_layers=num_layers)
        self.device = device

    def forward(self, src, tgt):
        src = self.positional_encoding(src)
        memory = self.encoder.forward(src)
        tgt = self.positional_encoding(tgt)
        output = self.decoder.forward(tgt, memory)
        return output
    
class SPARTA_H (nn.Module):
    def __init__(self, d_model, num_heads, d_feedforward, num_layers, max_len=1000, device='cuda:0', dropout=0.1):
        super(SPARTA_H, self).__init__()
        self.CTD = SPARTA_C(d_model, num_heads, d_feedforward, num_layers, max_len, device, dropout)
        self.FTD = SPARTA_F(d_model, num_heads, d_feedforward, num_layers, max_len, device)
    
    def forward(self, src, tgt):
        CTD_output = self.CTD(src, src)
        FTD_output = self.FTD(src, tgt)
        return CTD_output, FTD_output