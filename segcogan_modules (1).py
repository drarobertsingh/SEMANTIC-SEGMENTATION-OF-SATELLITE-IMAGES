import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------
# Generator - U-Net Architecture
# ------------------------------------------
class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_dropout=False):
        super(UNetBlock, self).__init__()
        if down:
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True)
            )
        else:
            self.block = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        self.use_dropout = use_dropout
        if self.use_dropout:
            self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.block(x)
        return self.dropout(x) if self.use_dropout else x

class GeneratorUNet(nn.Module):
    def __init__(self, input_nc, output_nc):
        super(GeneratorUNet, self).__init__()

        # Encoder
        self.enc1 = UNetBlock(input_nc, 64, down=True, use_dropout=False)   # 256 -> 128
        self.enc2 = UNetBlock(64, 128, down=True)
        self.enc3 = UNetBlock(128, 256, down=True)
        self.enc4 = UNetBlock(256, 512, down=True)
        self.enc5 = UNetBlock(512, 512, down=True)

        # Decoder
        self.dec1 = UNetBlock(512, 512, down=False, use_dropout=True)
        self.dec2 = UNetBlock(1024, 256, down=False, use_dropout=True)
        self.dec3 = UNetBlock(512, 128, down=False)
        self.dec4 = UNetBlock(256, 64, down=False)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, output_nc, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)

        d1 = self.dec1(e5)
        d2 = self.dec2(torch.cat([d1, e4], dim=1))
        d3 = self.dec3(torch.cat([d2, e3], dim=1))
        d4 = self.dec4(torch.cat([d3, e2], dim=1))
        out = self.final(torch.cat([d4, e1], dim=1))

        return out


# ------------------------------------------
# Discriminator - PatchGAN
# ------------------------------------------
class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(input_nc * 2, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 1, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 4, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        # Concatenate input and target images
        inp = torch.cat([x, y], dim=1)
        return self.model(inp)


# ------------------------------------------
# Loss Functions
# ------------------------------------------
def adversarial_loss(pred, target):
    return F.binary_cross_entropy(pred, target)

def segmentation_loss(pred, target):
    return F.cross_entropy(pred, target)

def dice_loss(pred, target):
    smooth = 1.
    iflat = pred.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    return 1 - ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))


# ------------------------------------------
# Evaluation Metrics
# ------------------------------------------
def compute_iou(pred, target):
    pred = pred.argmax(dim=1)
    intersection = (pred & target).float().sum((1, 2))
    union = (pred | target).float().sum((1, 2))
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean()

def compute_pixel_accuracy(pred, target):
    pred = pred.argmax(dim=1)
    correct = (pred == target).float()
    return correct.sum() / correct.numel()


def compute_dice(pred, target):
    pred = pred.argmax(dim=1)
    smooth = 1e-6
    intersection = (pred * target).sum().float()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
