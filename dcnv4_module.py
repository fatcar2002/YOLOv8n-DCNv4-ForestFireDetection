import torch
import torch.nn as nn
from DCNv4 import DCNv4

class DCNv4_2D(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.c2 = c2
        self.stride = s
        
        # 1. Ensure channels are divisible by 16 for DCNv4
        c2_dcn = c2 if c2 % 16 == 0 else ((c2 // 16) + 1) * 16
        group = max(1, c2_dcn // 16 if c2_dcn % 32 != 0 else c2_dcn // 32)

        # 2. If stride > 1, conv1 acts as a strided conv for spatial downsampling.
        # If stride == 1, conv1 acts as a 1x1 channel adapter.
        if s > 1:
            self.conv1 = nn.Conv2d(c1, c2_dcn, kernel_size=k, stride=s, padding=p if p is not None else k // 2)
        else:
            self.conv1 = nn.Conv2d(c1, c2_dcn, kernel_size=1, stride=1) if c1 != c2_dcn else nn.Identity()
        
        # 3. DCNv4 core operator runs at stride=1 internally to match sequence length L
        self.dcnv4 = DCNv4(
            channels=c2_dcn,
            kernel_size=k,
            stride=1,
            pad=p if p is not None else k // 2,
            group=group,
            without_pointwise=True
        )
        
        # 4. Project back to target c2 channels
        self.proj_out = nn.Conv2d(c2_dcn, c2, 1)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        orig_device = x.device
        if not x.is_cuda:
            x = x.cuda()
            self.cuda()

        # Step 1: conv1 handles channel alignment and native downsampling (if stride > 1)
        x = self.conv1(x)
        B, C, H, W = x.shape
        
        # Step 2: DCNv4 operates on the feature map
        x_flat = x.permute(0, 2, 3, 1).contiguous().view(B, H * W, C).contiguous()
        out_flat = self.dcnv4(x_flat, (int(H), int(W)))
        
        # Step 3: Reshape back to 4D tensor
        if out_flat.dim() == 4:
            out = out_flat.permute(0, 3, 1, 2).contiguous()
        else:
            out = out_flat.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            
        # Step 4: Project channels to target c2 and activate
        out = self.proj_out(out)
        out = self.act(out)

        if orig_device.type == 'cpu':
            out = out.to(orig_device)

        return out