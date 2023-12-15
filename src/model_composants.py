from utils import *
import numpy as np
import math

class DepthWise_Conv2d(nn.Module):
    def __init__(self, in_channels, kernel_size, stride, padding, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=in_channels,
                              kernel_size=kernel_size,
                              stride = stride,
                              padding = padding,
                              groups=in_channels,
                              bias=bias)
        
    def forward(self, x):
        return self.conv(x)
    
class PointWise_Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=1,
                              stride = 1,
                              bias=bias)
        
    def forward(self, x):
        return self.conv(x)

class Rwin_SA(nn.Module):
    def __init__(self,
                C = 180,
                window_size = (8, 32),
                nb_heads = 6,
                attention_dropout = 0.,
                qkv_bias = True):
        """ https://arxiv.org/pdf/2211.13654.pdf
        Compute Local Self-Attention with window mechanism using Dynamic Position Bias
        Input:
            C                   : int, size of embedding dimension
            window_size         : tuple(int, int) size of sliding window to split images
            nb_heads            : int, number of heads for Attention
            attention_dropout   : float, drop out rate for attention
            qkv_bias            : bool, use bias for QKV
        """
        super().__init__()
        self.window_size = window_size
        h, w = window_size
        self.nb_heads = nb_heads
        self.C = C
        assert C % nb_heads == 0, "C must be divisible by number of heads"
        self.head_dim = C // nb_heads
        self.scale = self.head_dim ** -0.5
        # QKV
        self.QKV  = nn.Linear(C, 3*C, bias=qkv_bias)
        # Final projection
        self.final_projection = nn.Linear(C, C)
        # Attention drop out
        self.attention_dropout = nn.Dropout(attention_dropout)
        # DPB
        self.pos_bias = Dynamic_Position_Bias(C//4, nb_heads)
        # generate mother-set for biases
        position_bias_h = torch.arange(1 - h, h) # (2*h-1)
        position_bias_w = torch.arange(1 - w, w) # (2*w-1)
        biases = torch.stack(torch.meshgrid([position_bias_h, position_bias_w])) # (2, 2*h-1, 2*w-1)
        biases = biases.flatten(1).transpose(0, 1).contiguous().float() # (2, [2*h-1]*[2*w-1]) -> ([2*h-1]*[2*w-1]), 2)
        self.register_buffer('rpe_biases', biases) # auto set to device of model

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(h) # h
        coords_w = torch.arange(w) # w
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # (2, h, w)
        coords_flatten = torch.flatten(coords, 1)  # (2, h*w)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # (2, h*w, h*w)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # (h*w, h*w, 2)
        relative_coords[:, :, 0] += h - 1  # shift to start from 0
        relative_coords[:, :, 1] += w - 1
        # to distinguish pixels, exp: for top-left corner pixel, left and down are same distance -> must have different value
        relative_coords[:, :, 0] *= 2 * w - 1
        relative_position_index = relative_coords.sum(-1)  # (h*w, h*w)
        self.register_buffer("relative_position_index", relative_position_index)
        
        
    def forward(self, X):
        """
        Input:
            X           : 4-D Tensor (Batch, C, H, W)
        Output:
            attention   : Local self-attention as 4-D Tensor (Batch, C, H, W)
        """
        H, W = X.size(2), X.size(3)
        # split image into non-overlaped patches
        X_splitted = split_images(X, self.window_size) # (Batch*NB_patches, h*w, C) with (h, w): window_size
        B_nbPatches, N, C = X_splitted.size() # N = h*w = number of pixels in a window

        qkv = self.QKV(X_splitted) # (Batch*NB_patches, N, 3*C)
        qkv = torch.reshape(qkv, (B_nbPatches, N, 3, C)) # (Batch*NB_patches, N, 3, C)
        qkv = torch.reshape(qkv, (B_nbPatches, N, 3, self.nb_heads, C//self.nb_heads)) # (Batch*NB_patches, N, 3, nb_heads, d)
        qkv = qkv.permute(2, 0, 3, 1, 4) # (3, Batch*NB_patches, nb_heads, N, d)
        q, k, v = qkv[0], qkv[1], qkv[2] # each (Batch*NB_patches, nb_heads, N, d)
        
        # self-attention
        attention = (q @ k.transpose(-2, -1)) * self.scale # (Batch*NB_patches, nb_heads, N, N)
        # get relative position bias
        pos_table = self.pos_bias(self.rpe_biases)
        rpb = pos_table[self.relative_position_index.view(-1)] # (h*w*h*w, nb_heads) or (N*N, nb_heads)
        rpb = rpb.view(N, N, self.nb_heads) # (N, N, nb_heads)
        rpb = rpb.permute(2, 0, 1).contiguous() # (nb_heads, N, N)
        rpb = rpb.unsqueeze(0) # add batch dim, (1, nb_heads, N, N), same dim as attention
        attention = attention + rpb # (Batch*NB_patches, nb_heads, N, N)

        attention = torch.softmax(attention, dim=-1)
        attention = attention @ v # (Batch*NB_patches, nb_heads, N, d)
        attention = self.attention_dropout(attention)

        attention = attention.transpose(1, 2) # (Batch*NB_patches, N, nb_heads, d)
        attention = attention.reshape(-1, N, C) # (Batch*NB_patches, N, C)

        x = self.final_projection(attention)

        x = merge_splitted_images(x, self.window_size, (H, W))  # (Batch, C, H, W)
        return x
    
class Dynamic_Position_Bias(nn.Module):
    """
    https://arxiv.org/pdf/2303.06908.pdf
    https://arxiv.org/pdf/2103.14030.pdf
    Objective is to generate the relative position bias dynamically
    """
    def __init__(self, dim, nb_heads):
        """
        Input:
            dim     : intermediate layers's dimension, set as D/4 as suggested the authors, D is embedding dimension
            nb_heads: number of heads of MHA
        """
        super().__init__()
        self.nb_heads = nb_heads
        self.dpe_dim = dim//4
        self.seq = nn.Sequential(
            # 1st block, input is 2 as relative distance of x and y
            nn.Linear(2, self.dpe_dim),
            nn.LayerNorm(self.dpe_dim),
            nn.ReLU(),
            # 2nd block
            nn.Linear(self.dpe_dim, self.dpe_dim),
            nn.LayerNorm(self.dpe_dim),
            nn.ReLU(),
            # 3rd block
            nn.Linear(self.dpe_dim, self.dpe_dim),
            nn.LayerNorm(self.dpe_dim),
            nn.ReLU(),
            # final projection, on paper dim_out is 1, but avoid doing all over again for each head
            nn.Linear(self.dpe_dim, self.nb_heads),
        )

    def forward(self, biases):
        """
        Input
            biases  : 2-D Tensor (h*w, 2)
        Output
            out     : 2-D Tensor (h*w, nb_heads)
        """
        return self.seq(biases)
    
    
class L_SA_block(nn.Module):
    """
    L_SA_block
    """
    def __init__(self, C, hidden_ratio, nb_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(C)
        self.lsa = Rwin_SA(nb_heads = nb_heads)
        self.ln2 = nn.LayerNorm(C)
        self.mlp = MLP(C, hidden_ratio)

    def forward(self, x):
        """
        Input:
            x   : 4-D tensor (B, C, H, W)
        """
        # LN
        in_ln1 = x.permute(0, 2, 3, 1) # (B, H, W, C)
        out_ln1 = self.ln1(in_ln1) # (B, H, W, C)
        # LSA
        in_lsa = out_ln1.permute(0, 3, 1, 2) # (B, C, H, W)
        out_lsa = self.lsa(in_lsa) # (B, C, H, W)
        # LN
        in_ln2 = out_lsa + x # (B, C, H, W)
        in_ln2 = in_ln2.permute(0, 2, 3, 1) # (B, H, W, C)
        out_ln2 = self.ln2(in_ln2) # (B, H, W, C)
        # MLP
        out_mlp = self.mlp(out_ln2) # (B, H, W, C)
        out_mlp = out_mlp.permute(0, 3, 1, 2) # (B, C, H, W)
        out = out_mlp + in_ln2.permute(0, 3, 1, 2) # (B, C, H, W)
        return out
         
class RG_SA(nn.Module):
    """
    Recursive-generalization self-attention
    """
    def __init__(self, C, cr, s_r, nb_heads):
        super().__init__()
        self.s_r = s_r
        Cr = int(C*cr)
        self.nb_heads = nb_heads
        self.head_q_k_dim = Cr//nb_heads
        self.head_v_dim = C//nb_heads
        self.scale = self.head_q_k_dim**-0.5
        self.Q = nn.Linear(C, Cr, bias=True)
        self.K = nn.Linear(Cr, Cr, bias=True)
        self.V = nn.Linear(Cr, C, bias=True)
        
        # Depth-wise convolutional layer for the recursion
        self.DWConv = DepthWise_Conv2d(in_channels=C, kernel_size=4, stride = s_r, padding = 0, bias=False)
        # Depth-wise 3x3 convolutional layer
        self.DWConv2 = DepthWise_Conv2d(in_channels=180, kernel_size=3, stride = 1, padding = 1, bias=False)
        # Point-wise 1x1 convolutional layer
        self.PWConv = PointWise_Conv2d(in_channels=C, out_channels=Cr, bias=False)
        

    def forward(self, x, h):
        B, C, H, W = x.size()
        # Q
        q = x.view(B, C, H*W).permute(0, 2, 1) # (B, H*W, C)
        q = self.Q(q) # (B, H*W, Cr)
        q = q.view(B, H*W, self.nb_heads, self.head_q_k_dim) # (B, H*W, nb_head, d_q_k)
        q = q.permute(0, 2, 1, 3).contiguous() # (B, nb_head, H*W, d_q_k)
        T = int(math.log(H/h, self.s_r))
        
        for i in range(T):
            if i == 0:
                x_r = self.DWConv(x)
            else:
                x_r = self.DWConv(x_r) # (B, C, h, w)
        x_r = self.DWConv2(x_r) # (B, Cr, h, w)
        x_r = self.PWConv(x_r) # (B, Cr, h, w)
        x_r = x_r.view(B, x_r.size(1), -1) # (B, Cr, h*w)
        x_r = x_r.permute(0, 2, 1).contiguous() # (B, h*w, Cr)
        # K
        k = self.K(x_r) # (B, h*w, Cr)
        k = k.view(B, k.size(1), self.nb_heads, self.head_q_k_dim) # (B, h*w, nb_heads, d_q_k)
        k = k.permute(0, 2, 1, 3).contiguous() # (B, nb_head, h*w, d_q_k)
        # V
        v = self.V(x_r) # (B, h*w, C)
        v = v.view(B, v.size(1), self.nb_heads, self.head_v_dim) # (B, h*w, nb_head, d_v)
        v = v.permute(0, 2, 1, 3).contiguous() # (B, nb_head, h*w, d_v)

        attention = (q @ k.transpose(-2, -1)) * self.scale # (B, nb_head, H*W, h*w)
        attention = torch.softmax(attention, dim=-1) # (B, nb_head, H*W, h*w)
        
        x = attention @ v # (B, nb_head, H*W, d_v)
        x = x.permute(0, 1, 3, 2).contiguous() # (B, nb_head, d_v, H*W)
        x = x.view(B, C, H*W) # (B, C, H*W) nb_head * d_v = C
        x = x.view(B, C, H, W) # (B, C,H, W)
        
        return x

    
class RG_SA_block(nn.Module):
    """
    RG_SA_block
    """
    def __init__(self, C, cr, s_r, hidden_ratio, nb_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(C)
        self.rg_sa = RG_SA(C, cr, s_r, nb_heads)
        self.ln2 = nn.LayerNorm(C)
        self.mlp = MLP(C, hidden_ratio)

    def forward(self, x, h):
        """
        Input:
            x   : 4-D tensor (B, C, H, W)
        """
        # LN
        in_ln1 = x.permute(0, 2, 3, 1) # (B, H, W, C)
        out_ln1 = self.ln1(in_ln1) # (B, H, W, C)
        # RG-SA
        in_rgsa = out_ln1.permute(0, 3, 1, 2) # (B, C, H, W)
        out_rgsa = self.rg_sa(in_rgsa, h) # (B, C, H, W)
        # LN
        in_ln2 = out_rgsa + x # (B, C, H, W)
        in_ln2 = in_ln2.permute(0, 2, 3, 1) # (B, H, W, C)
        out_ln2 = self.ln2(in_ln2) # (B, H, W, C)
        # MLP
        out_mlp = self.mlp(out_ln2) # (B, H, W, C)
        out_mlp = out_mlp.permute(0, 3, 1, 2) # (B, C, H, W)
        out = out_mlp + in_ln2.permute(0, 3, 1, 2) # (B, C, H, W)
        return out
         
class RG(nn.Module):
    """
    Residual Group
    """
    def __init__(self, N2, C, cr, s_r, hidden_ratio, nb_heads):
        super().__init__()
        self.N2 = N2
        self.HAIs = nn.ModuleList([HAL(C) for _ in range(2*N2)])
        self.LSAs = nn.ModuleList([L_SA_block(C, hidden_ratio, nb_heads) for _ in range(N2)])
        self.RGSAs = nn.ModuleList([RG_SA_block(C, cr, s_r, hidden_ratio, nb_heads) for _ in range(N2)])
        self.final_conv = nn.Conv2d(C, C, kernel_size = 3, stride = 1, padding = 1)

    def forward(self, x, h):
        x_ = x # (B, C, H, W)
        for i in range(self.N2):
            # LSA
            hai = self.HAIs[2*i](x_) # (B, C, H, W)
            x_ = self.LSAs[i](x_) # (B, C, H, W)
            x_ = hai + x_ # (B, C, H, W)
            # RGSA
            hai2 = self.HAIs[2*i+1](x_) # (B, C, H, W)
            x_ = self.RGSAs[i](x_, h) # (B, C, H, W)
            x_ = x_ + hai2 # (B, C, H, W)
        return self.final_conv(x_) + x
    
class HAL(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.alpha = nn.Parameter(torch.randn(1, C, 1, 1))
        
    def forward(self, x):
        x = x * self.alpha
        return x

class Reconstruction_Module(nn.Module):
    """
    Reconstruction module using PixelShuffle as upsampler
    read -> params
    https://arxiv.org/pdf/1609.05158.pdf
    """
    def __init__(self, in_channels, out_channels, upscale_factor, stride_r):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = in_channels,
                             out_channels = in_channels * (upscale_factor**2),
                             kernel_size = 3,
                             stride = 1,
                             padding = 1
                            )
        self.pixelShuffle = nn.PixelShuffle(upscale_factor=upscale_factor)
        self.conv2 = nn.Conv2d(in_channels = in_channels,
                             out_channels = out_channels,
                             kernel_size = 3,
                             stride = 1,
                             padding = 1
                            )
        
    def forward(self, x):
        out_conv1 = self.conv1(x)
        out_shuffled = self.pixelShuffle(out_conv1)
        out_conv2 = self.conv2(out_shuffled)
        return out_conv2
    
class MLP(nn.Module):
    def __init__(self, dim_in, hidden_ratio):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(dim_in, hidden_ratio*dim_in, bias=True),
            nn.Linear(hidden_ratio*dim_in, dim_in, bias=True)
        )

    def forward(self, X):
        """
        MLP on dim C
        Input:
            X   : 4-D Tensor (Batch, C, H, W)
        Output:
            out : 4-D Tensor (Batch, C, H, W)
        """
        return self.seq(X)

# Unit tests here
if __name__ == "__main__":
    img_size = 64
    print("Tests")
    print("DynamicPositionBias")
    batch_size = 2
    dim = 180
    nb_heads = 6
    h_window = 8
    w_window = 32
    biases = torch.rand((h_window*w_window, 2))
    X = torch.rand((batch_size, dim, img_size, img_size)) # (B, C, H, W)
    dpb = Dynamic_Position_Bias(dim, nb_heads)
    assert dpb(biases).shape == (h_window*w_window, nb_heads), "Error DPB"
    print("DynamicPositionBias: done")
    print("Rwin_SA")
    rwin = Rwin_SA(C = dim, window_size=(h_window, w_window), nb_heads=nb_heads)
    assert rwin(X).shape == (batch_size, dim, img_size, img_size), "Error Rwin"
    print("Rwin_SA: done")
    print("MLP")
    mlp = MLP(dim, hidden_ratio = 2)
    assert mlp(X).shape == (batch_size, img_size, img_size, dim), "Error MLP"
    print("RG_SA")
    rg_sa = RG_SA(...)
    assert rg_sa(X).shape == (batch_size, dim, img_size, img_size), "Error RG_SA"

