import torch
import torch.nn as nn
import torch.nn.functional as F

class Convs1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),  # Add padding
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),  # Add padding
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)

class Downsampling(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.maxpool(x)

class Upsampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsampling = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        return self.upsampling(x)

class Unet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.downsample = Downsampling()
        self.block1 = Convs1(1, 64)
        self.block2 = Convs1(64, 128)
        self.block3 = Convs1(128, 256)
        self.block4 = Convs1(256, 512)
        self.block5 = Convs1(512, 1024)
        self.upsample1 = Upsampling(1024, 512)
        self.block6 = Convs1(1024, 512)
        self.upsample2 = Upsampling(512, 256)
        self.block7 = Convs1(512, 256)
        self.upsample3 = Upsampling(256, 128)
        self.block8 = Convs1(256, 128)
        self.upsample4 = Upsampling(128, 64)
        self.block9 = Convs1(128, 64)
        self.conv10 = nn.Conv2d(64, num_classes, kernel_size=1)
    

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(self.downsample(x1))
    
        x3 = self.block3(self.downsample(x2))

        x4 = self.block4(self.downsample(x3))
  
        x5 = self.block5(self.downsample(x4))
        x5 = self.upsample1(x5)
        x4 = F.interpolate(x4, size=x5.shape[-2:], mode="bilinear", align_corners=False)
        x6 = self.block6(torch.cat((x5, x4), dim=1))
        
        x6 = self.upsample2(x6)
        x3 = F.interpolate(x3, size=x6.shape[-2:], mode="bilinear", align_corners=False)
        x7 = self.block7(torch.cat((x6, x3), dim=1))
        
        x7 = self.upsample3(x7)
        x2 = F.interpolate(x2, size=x7.shape[-2:], mode="bilinear", align_corners=False)
        x8 = self.block8(torch.cat((x7, x2), dim=1))
        
        x8 = self.upsample4(x8)
        x1 = F.interpolate(x1, size=x8.shape[-2:], mode="bilinear", align_corners=False)
        x9 = self.block9(torch.cat((x8, x1), dim=1))
        
        x10 = self.conv10(x9)
        return x10
    
    def load_model(self, path):
        self.load_state_dict(torch.load(path))

    def save_model(self, path):
        torch.save(self.state_dict(), path)



# Test
if __name__ == "__main__":
    model = Unet(num_classes=2)
    x = torch.randn(3, 1, 128, 128)
    output = model(x)
    print(output.shape)  # Should be [3, 2, 128, 8]