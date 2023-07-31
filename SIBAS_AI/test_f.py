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
import scipy.ndimage as ndi

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
    def __init__(self, csv_file, transform=None, infer=False):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.infer = infer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 1]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.infer:
            if self.transform:
                image = self.transform(image=image)['image']
            return image

        mask_rle = self.data.iloc[idx, 2]
        mask = rle_decode(mask_rle, (image.shape[0], image.shape[1]))

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask

test_transform = A.Compose(
    [   
        #A.Resize(224, 224),
        A.Normalize(),
        ToTensorV2()
    ]
)

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

# Dice Coefficient 계산 함수
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred)
    return (2.0 * intersection + smooth) / (union + smooth)

def morphological_operations(mask):
    # 침식 연산을 통해 객체 크기 축소
    kernel_erosion = np.ones((3, 3), np.uint8)
    eroded_mask = cv2.erode(mask, kernel_erosion, iterations=1)

    # 팽창 연산을 통해 객체 크기 확장
    kernel_dilation = np.ones((3, 3), np.uint8)
    dilated_mask = cv2.dilate(eroded_mask, kernel_dilation, iterations=1)
    
    # 침식 연산을 통해 객체 크기 축소
    #kernel_erosion2 = np.ones((2, 2), np.uint8)
    #eroded_mask2 = cv2.erode(dilated_mask, kernel_erosion2, iterations=1)
#
    ## 팽창 연산을 통해 객체 크기 확장
    #kernel_dilation2 = np.ones((2, 2), np.uint8)
    #dilated_mask2 = cv2.dilate(eroded_mask2, kernel_dilation2, iterations=1)
#
    ## 침식 연산을 통해 객체 크기 축소
    #kernel_erosion3 = np.ones((2, 2), np.uint8)
    #eroded_mask3 = cv2.erode(dilated_mask2, kernel_erosion3, iterations=1)
#
    ## 팽창 연산을 통해 객체 크기 확장
    #kernel_dilation3 = np.ones((2, 2), np.uint8)
    #dilated_mask3 = cv2.dilate(eroded_mask3, kernel_dilation3, iterations=2)
#
    return dilated_mask

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = A2FPN(3).to(device)

    # Load the previously saved model if available
    saved_model_path = './best_model.pt'
    if os.path.exists(saved_model_path):
        model.load_state_dict(torch.load(saved_model_path))
        print("Loaded previously saved model.")

    test_dataset = SatelliteDataset(csv_file='./test.csv', transform=test_transform, infer=True)
    test_dataloader = DataLoader(test_dataset, batch_size=24, shuffle=False, num_workers=4)

    with torch.no_grad():
        model.eval()
        result = []
        for images in tqdm(test_dataloader):
            images = images.float().to(device)

            outputs = model(images)
            masks = torch.sigmoid(outputs).cpu().numpy()
            masks = np.squeeze(masks, axis=1)
            masks = (masks > 0.35).astype(np.uint8)  # Threshold = 0.2745

            for i in range(len(images)):
                mask_rle = rle_encode(masks[i])
                if mask_rle == '':
                    result.append(-1)
                else:
                    mask_binary = (masks[i] > 0.35).astype(np.uint8)
                    mask_morph = morphological_operations(mask_binary)
                    mask_rle = rle_encode(mask_morph)
                    if mask_rle == '':
                        result.append(-1)
                    else:
                        result.append(mask_rle)

    submit = pd.read_csv('./sample_submission.csv')
    submit['mask_rle'] = result

    submit.to_csv('./submit4.csv', index=False)

if __name__ == "__main__":
    main()