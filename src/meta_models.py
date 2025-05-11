import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


# ===== Simple UNet =====
class SimpleUNet(nn.Module):
    def __init__(self, num_experts, in_channels=3):
        super(SimpleUNet, self).__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(in_channels, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.Conv2d(128, 128, 3, padding=1), nn.ReLU())
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.Conv2d(256, 256, 3, padding=1), nn.ReLU())

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(), nn.Conv2d(128, 128, 3, padding=1), nn.ReLU())

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())

        self.final = nn.Conv2d(64, num_experts, 1)

        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')


    def center_crop(self, tensor, target_tensor):
        _, _, h, w = target_tensor.shape
        tensor = torchvision.transforms.functional.center_crop(tensor, [h, w])
        return tensor

    def forward(self, x):
        w_pad = (x.shape[-1] % 4) // 2
        h_pad = (x.shape[-2] % 4) // 2
        x_pad = F.pad(x, (w_pad, w_pad, h_pad, h_pad), mode='replicate')  # Pad to make it divisible by 4
        e1 = self.enc1(x_pad)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        b = self.bottleneck(p2)
        
        u2 = self.up2(b)
        u2 = torch.cat([u2, e2], dim=1)
        d2 = self.dec2(u2)

        u1 = self.up1(d2)
        u1 = torch.cat([u1, e1], dim=1)
        d1 = self.dec1(u1)

        out = self.final(d1)
        return self.center_crop(out, x)


class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)


    def center_crop(self, tensor, target_tensor):
        _, _, h, w = target_tensor.size()
        tensor = torchvision.transforms.functional.center_crop(tensor, [h, w])
        return tensor
    
    def forward(self, g, x):
        # Crop if shapes don't match
        if g.size() != x.size():
            g = self.center_crop(g, x)
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

# ===== Simple Attention UNet =====
class AttentionUNet(nn.Module):
    def __init__(self, num_experts, in_channels=3):
        super(AttentionUNet, self).__init__()
        self.enc1 = self.contracting_block(in_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.enc2 = self.contracting_block(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.bottleneck = self.contracting_block(128, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.att2 = AttentionBlock(F_g=128, F_l=128, F_int=64)
        self.dec2 = self.contracting_block(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.att1 = AttentionBlock(F_g=64, F_l=64, F_int=32)
        self.dec1 = self.contracting_block(128, 64)

        self.final = nn.Conv2d(64, num_experts, kernel_size=1)


        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')


    def center_crop(self, tensor, target_tensor):
        _, _, h, w = target_tensor.size()
        tensor = torchvision.transforms.functional.center_crop(tensor, [h, w])
        return tensor


    def contracting_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        b = self.bottleneck(p2)

        # Decoder 2
        up2 = self.up2(b)
        e2 = self.att2(g=up2, x=e2)
        up2 = self.center_crop(up2, e2)
        d2 = self.dec2(torch.cat([up2, e2], dim=1))

        # Decoder 1
        up1 = self.up1(d2)
        e1 = self.att1(g=up1, x=e1)
        up1 = self.center_crop(up1, e1)
        d1 = self.dec1(torch.cat([up1, e1], dim=1))

        out = self.final(d1)
        return out