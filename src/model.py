from model_composants import *

class RGT(nn.Module):
    def __init__(self, C, dim_in, N1, N2, cr, s_r, hidden_ratio, nb_heads, upscale_factor):
        super().__init__()
        self.C = C
        self.s_r = s_r
        # F_0
        # conv
        # in: 3 x H x W -> out: C x H x W
        self.conv1 = nn.Conv2d(in_channels=dim_in,
                               out_channels=C,
                               kernel_size = 3,
                               stride=1,
                               padding=1)
        # F_1 -> F_d-1
        # residual groups (RGs)
        # in: C x H x W -> out: C x H x W
        RGs = []
        for _ in range(N1):
            RGs.append(RG(N2, C, cr, s_r, hidden_ratio, nb_heads))
        self.RGs = nn.ModuleList(RGs)
        # F_d-1 -> F_d
        # conv
        # in: C x H x W -> out: C x H x W
        self.conv2 = nn.Conv2d(in_channels=C,
                               out_channels=C,
                               kernel_size = 3,
                               stride=1,
                               padding=1)
        # reconstruction module (pixel-shuffle and convolutional layers)
        # in: C x H x W -> out: 3 x H_hat x W_hat
        self.reconstruction = Reconstruction_Module(in_channels=C,
                                                    out_channels=dim_in,
                                                    upscale_factor = upscale_factor,
                                                    stride_r=s_r)
        
    def forward(self, x, h):
        """
        Input
            x    : 4-D tensor (B, dim_in, H, W)
        """
        x = self.conv1(x)
        for i in range(len(self.RGs)):
            if i == 0:
              x_ = self.RGs[i](x, h)
            else:
              x_ = self.RGs[i](x_, h)
        x_ = self.conv2(x_)
        x = x + x_
        x = self.reconstruction(x)
        return x
