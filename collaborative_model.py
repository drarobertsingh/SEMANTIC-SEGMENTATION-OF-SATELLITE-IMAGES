import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, in_channels):
        super(Encoder, self).__init__()
        self.enc1 = ConvBlock(in_channels, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        return x1, x2, x3


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(128, 64)
        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x3, x2, x1):
        x = self.up1(x3)
        x = self.dec1(torch.cat([x, x2], dim=1))
        x = self.up2(x)
        x = self.dec2(torch.cat([x, x1], dim=1))
        x = self.final(x)
        return x


class CollaborativeModel(nn.Module):
    def __init__(self, in_channels=3):
        super(CollaborativeModel, self).__init__()
        self.encoder = Encoder(in_channels)
        self.decoder1 = Decoder()
        self.decoder2 = Decoder()

    def forward(self, x):
        x1, x2, x3 = self.encoder(x)
        out1 = self.decoder1(x3, x2, x1)
        out2 = self.decoder2(x3, x2, x1)
        return out1, out2


def collaborative_loss(pred1, pred2, target):
    loss_fn = nn.BCEWithLogitsLoss()
    loss1 = loss_fn(pred1, target)
    loss2 = loss_fn(pred2, target)
    consistency = F.mse_loss(torch.sigmoid(pred1), torch.sigmoid(pred2))
    return loss1 + loss2 + 0.1 * consistency


def compute_metrics(pred, target):
    pred = (torch.sigmoid(pred) > 0.5).float()
    target = target.float()

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = intersection / (union + 1e-6)
    dice = 2 * intersection / (pred.sum() + target.sum() + 1e-6)
    pixel_acc = (pred == target).float().mean()
    return {
        'IoU': iou.item(),
        'Dice': dice.item(),
        'Pixel Accuracy': pixel_acc.item()
    }


if __name__ == "__main__":
    model = CollaborativeModel()
    dummy_input = torch.randn(1, 3, 256, 256)
    target = torch.randint(0, 2, (1, 1, 256, 256)).float()

    out1, out2 = model(dummy_input)
    loss = collaborative_loss(out1, out2, target)
    metrics = compute_metrics(out1, target)

    print("Loss:", loss.item())
    print("Metrics:", metrics)
