"""TCN architectures used by the repaired project.

`TCNCompact` is the CPU-budgeted architecture used for the controlled
MLP-vs-TCN experiment.  It has four causal residual blocks with dilations
1,2,4,8 and kernel size 3, hence receptive field
    1 + 2*(1+2+4+8) = 31 hours,
which covers the full 24-hour input.

`TCNFull` is a larger two-convolution-per-block variant with a 61-hour
receptive field.  It is provided for future longer training but is not used
for the reported CPU-budgeted comparison.
"""
import torch
from torch import nn
import torch.nn.functional as F

class CausalConv1d(nn.Module):
    def __init__(self,c_in,c_out,kernel_size=3,dilation=1):
        super().__init__();self.left_pad=(kernel_size-1)*dilation;self.conv=nn.Conv1d(c_in,c_out,kernel_size,dilation=dilation)
    def forward(self,x):return self.conv(F.pad(x,(self.left_pad,0)))

class CompactBlock(nn.Module):
    def __init__(self,channels,dilation,dropout=.05):
        super().__init__();self.conv=CausalConv1d(channels,channels,3,dilation);self.drop=nn.Dropout(dropout)
    def forward(self,x):return F.relu(self.drop(self.conv(x))+x)

class TCNCompact(nn.Module):
    def __init__(self,input_channels,static_dim=4,hidden=16,output_dim=5,dropout=.05):
        super().__init__();self.input_proj=nn.Conv1d(input_channels,hidden,1);self.blocks=nn.Sequential(*[CompactBlock(hidden,d,dropout) for d in (1,2,4,8)]);self.head=nn.Sequential(nn.Linear(hidden+static_dim,32),nn.ReLU(),nn.Dropout(dropout),nn.Linear(32,output_dim))
    def forward(self,x,static):
        z=self.blocks(self.input_proj(x.transpose(1,2)));return self.head(torch.cat([z[:,:,-1],static],1))

class FullBlock(nn.Module):
    def __init__(self,channels,dilation,dropout=.1):
        super().__init__();self.c1=CausalConv1d(channels,channels,3,dilation);self.c2=CausalConv1d(channels,channels,3,dilation);self.n1=nn.GroupNorm(8,channels);self.n2=nn.GroupNorm(8,channels);self.drop=nn.Dropout(dropout)
    def forward(self,x):
        r=x;x=self.drop(F.gelu(self.n1(self.c1(x))));x=self.drop(self.n2(self.c2(x)));return F.gelu(x+r)

class TCNFull(nn.Module):
    def __init__(self,input_channels,static_dim=4,hidden=64,output_dim=5,dropout=.1):
        super().__init__();self.input_proj=nn.Conv1d(input_channels,hidden,1);self.blocks=nn.Sequential(*[FullBlock(hidden,d,dropout) for d in (1,2,4,8)]);self.head=nn.Sequential(nn.Linear(hidden+static_dim,64),nn.GELU(),nn.Dropout(dropout),nn.Linear(64,output_dim))
    def forward(self,x,static):
        z=self.blocks(self.input_proj(x.transpose(1,2)));return self.head(torch.cat([z[:,:,-1],static],1))
