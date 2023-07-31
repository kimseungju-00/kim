import os
import cv2
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torch.nn import Module, Conv2d, Parameter, Softmax

import albumentations as A
from albumentations.pytorch import ToTensorV2

from tqdm import tqdm
from sklearn.model_selection import train_test_split

# RLE decoding function
def rle_decode(mask_rle, shape):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)


# RLE encoding function
def rle_encode(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


# Custom Dataset
class SatelliteDataset(Dataset):
    def __init__(self, csv_file, augmentation=None, infer=False):
        self.data = pd.read_csv(csv_file)
        self.augmentation = augmentation
        self.infer = infer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # read data
        img_path = self.data.iloc[idx, 1]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.infer:
            if self.augmentation:
                image = self.augmentation(image=image)['image']
            return image

        mask_rle = self.data.iloc[idx, 2]
        mask = rle_decode(mask_rle, (image.shape[0], image.shape[1]))

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask


def get_training_augmentation():
    train_transform = [
        A.GaussNoise(p=0.2),
        A.OneOf(
            [
                A.RandomBrightnessContrast(p=1),
                A.RandomGamma(p=1),
                A.ChannelShuffle(always_apply=False, p=1.0)
            ],
            p=0.9,
        ),
        A.OneOf(
            [
                A.RandomBrightnessContrast(p=1),
                A.ChannelDropout(always_apply=False, p=1.0, channel_drop_range=(1, 1), fill_value=0),
                #A.Downscale(always_apply=False, p=0.2, scale_min=0.699999988079071, scale_max=0.9900000095367432, interpolation=2)
            ],
            p=0.9,
        ),
        A.OneOf(
            [
                #A.RandomBrightnessContrast(p=1),
                A.HueSaturationValue(p=1),
                A.CoarseDropout(always_apply=False, p=1.0, max_holes=8, max_height=8, max_width=8, min_holes=8, min_height=8, min_width=8)
            ],
            p=0.9,
        ),
        A.Normalize(),
        ToTensorV2()
    ]
    return A.Compose(train_transform)


def get_validation_augmentation():
    val_transform = [
        A.Normalize(),
        ToTensorV2()
    ]
    return A.Compose(val_transform)


def l2_norm(x):
    return torch.einsum("bcn, bn->bcn", x, 1 / torch.norm(x, p=2, dim=-2))


class ConvBnRelu(nn.Module):
    def __init__(self, in_planes, out_planes, ksize, stride, pad, dilation=1,
                 groups=1, has_bn=True, norm_layer=nn.BatchNorm2d, bn_eps=1e-5,
                 has_relu=True, inplace=True, has_bias=False):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=ksize,
                              stride=stride, padding=pad,
                              dilation=dilation, groups=groups, bias=has_bias)
        self.has_bn = has_bn
        if self.has_bn:
            self.bn = norm_layer(out_planes, eps=bn_eps)
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)

        return x


class Attention(Module):
    def __init__(self, in_places, scale=8, eps=1e-6):
        super(Attention, self).__init__()
        self.gamma = Parameter(torch.zeros(1))
        self.in_places = in_places
        self.l2_norm = l2_norm
        self.eps = eps

        self.query_conv = Conv2d(in_channels=in_places, out_channels=in_places // scale, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_places, out_channels=in_places // scale, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_places, out_channels=in_places, kernel_size=1)

    def forward(self, x):
        # Apply the feature map to the queries and keys
        batch_size, chnnels, width, height = x.shape
        Q = self.query_conv(x).view(batch_size, -1, width * height)
        K = self.key_conv(x).view(batch_size, -1, width * height)
        V = self.value_conv(x).view(batch_size, -1, width * height)

        Q = self.l2_norm(Q).permute(-3, -1, -2)
        K = self.l2_norm(K)

        tailor_sum = 1 / (width * height + torch.einsum("bnc, bc->bn", Q, torch.sum(K, dim=-1) + self.eps))
        value_sum = torch.einsum("bcn->bc", V).unsqueeze(-1)
        value_sum = value_sum.expand(-1, chnnels, width * height)

        matrix = torch.einsum('bmn, bcn->bmc', K, V)
        matrix_sum = value_sum + torch.einsum("bnm, bmc->bcn", Q, matrix)

        weight_value = torch.einsum("bcn, bn->bcn", matrix_sum, tailor_sum)
        weight_value = weight_value.view(batch_size, chnnels, height, width)

        return (self.gamma * weight_value).contiguous()


class AttentionAggregationModule(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(AttentionAggregationModule, self).__init__()
        self.convblk = ConvBnRelu(in_chan, out_chan, ksize=1, stride=1, pad=0)
        self.conv_atten = Attention(out_chan)

    def forward(self, s5, s4, s3, s2):
        fcat = torch.cat([s5, s4, s3, s2], dim=1)
        feat = self.convblk(fcat)
        atten = self.conv_atten(feat)
        feat_out = atten + feat
        return feat_out


class Conv3x3GNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=False):
        super().__init__()
        self.upsample = upsample
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3),
                      stride=1, padding=1, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.block(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return x


class FPNBlock(nn.Module):
    def __init__(self, pyramid_channels, skip_channels):
        super().__init__()
        self.skip_conv = nn.Conv2d(skip_channels, pyramid_channels, kernel_size=1)

    def forward(self, x):
        x, skip = x
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        skip = self.skip_conv(skip)

        x = x + skip
        return x


class SegmentationBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_upsamples=0):
        super().__init__()

        blocks = [
            Conv3x3GNReLU(in_channels, out_channels, upsample=bool(n_upsamples))
        ]

        if n_upsamples > 1:
            for _ in range(1, n_upsamples):
                blocks.append(Conv3x3GNReLU(out_channels, out_channels, upsample=True))

        self.block = nn.Sequential(*blocks)

    def forward(self, x):
        return self.block(x)


class A2FPN(nn.Module):
    def __init__(
            self,
            band,
            class_num=1,
            encoder_channels=[512, 256, 128, 64],
            pyramid_channels=64,
            segmentation_channels=64,
            dropout=0.2,
    ):
        super().__init__()
        self.base_model = models.resnet18(pretrained=True)
        self.base_layers = list(self.base_model.children())
        self.layer_down0 = nn.Sequential(*self.base_layers[:3])  # size=(N, 64, x.H/2, x.W/2)
        self.layer_down1 = nn.Sequential(*self.base_layers[3:5])  # size=(N, 64, x.H/4, x.W/4)
        self.layer_down2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer_down3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer_down4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)

        self.conv1 = nn.Conv2d(encoder_channels[0], pyramid_channels, kernel_size=(1, 1))

        self.p4 = FPNBlock(pyramid_channels, encoder_channels[1])
        self.p3 = FPNBlock(pyramid_channels, encoder_channels[2])
        self.p2 = FPNBlock(pyramid_channels, encoder_channels[3])

        self.s5 = SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=3)
        self.s4 = SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=2)
        self.s3 = SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=1)
        self.s2 = SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=0)

        self.attention = AttentionAggregationModule(segmentation_channels * 4, segmentation_channels * 4)
        self.final_conv = nn.Conv2d(segmentation_channels * 4, class_num, kernel_size=1, padding=0)
        self.dropout = nn.Dropout2d(p=dropout, inplace=True)

    def forward(self, x):
        c1 = self.layer_down0(x)
        c2 = self.layer_down1(c1)
        c3 = self.layer_down2(c2)
        c4 = self.layer_down3(c3)
        c5 = self.layer_down4(c4)

        p5 = self.conv1(c5)
        p4 = self.p4([p5, c4])
        p3 = self.p3([p4, c3])
        p2 = self.p2([p3, c2])

        s5 = self.s5(p5)
        s4 = self.s4(p4)
        s3 = self.s3(p3)
        s2 = self.s2(p2)

        out = self.dropout(self.attention(s5, s4, s3, s2))
        out = self.final_conv(out)
        out = F.interpolate(out, scale_factor=4, mode='bilinear', align_corners=True)

        return out

def calculate_dice(pred_mask, gt_mask):
    if np.sum(gt_mask) > 0 or np.sum(pred_mask) > 0:
        intersection = np.sum(pred_mask * gt_mask)
        return (2.0 * intersection + 1e-7) / (np.sum(pred_mask) + np.sum(gt_mask) + 1e-7)
    else:
        return None

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the CSV file containing image paths and mask RLEs
    dataset = pd.read_csv('./train.csv')

    # Split the dataset into training and validation sets (80% train, 20% val)
    train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)

    # Save the two datasets as separate CSV files
    train_data.to_csv('./train_split.csv', index=False)
    val_data.to_csv('./val_split.csv', index=False)

    # Create custom datasets for training and validation
    train_dataset = SatelliteDataset(csv_file='./train_split.csv', augmentation=get_training_augmentation())
    val_dataset = SatelliteDataset(csv_file='./val_split.csv', augmentation=get_validation_augmentation())

    train_loader = DataLoader(train_dataset, batch_size=6, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=6, shuffle=False, num_workers=4)

    model = A2FPN(3).to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)

    # Load the previously saved model if available
    saved_model_path = './best_model.pt'
    best_dice_score = 0.0
    if os.path.exists(saved_model_path):
        model.load_state_dict(torch.load(saved_model_path))
        print("Loaded previously saved model.")

# Training loop
    for epoch in range(50):
        model.train()
        epoch_loss = 0
        for images, masks in tqdm(train_loader):
            images = images.float().to(device)
            masks = masks.float().to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks.unsqueeze(1))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f'Epoch {epoch+1}, Loss: {epoch_loss/len(train_loader)}')

        # Validation
        model.eval()
        val_loss = 0
        val_dice_scores = []
        with torch.no_grad():
            for images, masks in tqdm(val_loader):
                images = images.float().to(device)
                masks = masks.float().to(device)

                outputs = model(images)
                loss = criterion(outputs, masks.unsqueeze(1))
                val_loss += loss.item()

                # Calculate Dice scores
                predicted_masks = torch.sigmoid(outputs)
                predicted_masks = (predicted_masks > 0.3).float()  # Threshold = 0.3
                predicted_masks = predicted_masks.cpu().numpy()
                masks = masks.cpu().numpy()
                for i in range(len(predicted_masks)):
                    dice_score = calculate_dice(predicted_masks[i], masks[i])
                    val_dice_scores.append(dice_score)

            avg_dice_score = np.mean(val_dice_scores)
            print(f'Validation Loss: {val_loss/len(val_loader)}, Dice Score: {avg_dice_score}')

            # Check if the current model is better than the previous best
            if avg_dice_score > best_dice_score:
                best_dice_score = avg_dice_score
                print("Saving the model...")
                torch.save(model.state_dict(), saved_model_path)
            
            scheduler.step()

if __name__ == "__main__":
    main()