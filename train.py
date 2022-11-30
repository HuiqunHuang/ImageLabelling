import typing as t

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from utils import DEVICE, synthesize_data
import torchvision.transforms as TR
### triplet loss pytorch for yaw
import torch
from PIL import Image
from torch import nn
# torch.set_grad_enabled(False)
from torchvision.models import resnet50
torch.set_grad_enabled(True)
import math
from collections import OrderedDict

class DETRdemo(nn.Module):
    """
    The model was built based on Demo DETR implementation from https://github.com/facebookresearch/detr. 1358.7476
    
    Only batch size 1 supported.
    
    Input: (Batch Size, 1, height, width)
    
    Output: [(Batch Size, x coordinate of the center, y coordinates of the center), (Batch Size, yaw), (Batch Size, width of the bounding box, height of the bounding box)]
    
    The outputs are all in the range of [0, 1], which in the relative original image coordinates in [xcenter, ycenter, w, h] format. The output yaw is in the relative range of [0, 2 * pi].
    
    To convert the output into the original range, we do the following calculations before the loss function:
    1) predicted xcenter * width of the original image; 2) predicted ycenter * height of the original image; 3) predicted yaw * 2 * pi; 4) predicted width * width of the original image; 5) predicted height * height of the original image;
    
    The height and width of the original image are the same and set as 200 in default.
    
    Loss function: loss function is MSE based. There are three main parts of the loss function, and the whole loss function can be represented by: 
    alpha1 * mse of the x y coordinates of the center + alph2 * yaw + alpha3 * mse of the width height of the bounding box,
    where alpha1, alpha2, and alpha3 are the coefficients to weight the importances of each part of the loss function. alpha1, alpha2, and alpha3 are set as 1, 10, and 1 in default as denoted in the variable of loss_fn_para.
    
    Main Structure of the model:
    1: ResNet-50. 
       To extract the spatial features of the image.
    2: Encoder of the Transformer, which was first proposed by the paper of "Attention is all you need".
       To 
    3: Decoder of the Transformer, which was first proposed by the paper of "Attention is all you need".
       To 
    """

    def __init__(self, num_classes=2, hidden_dim=128, nheads=8,
                 num_encoder_layers=6, num_decoder_layers=6):
        super(DETRdemo, self).__init__()

        # create ResNet-50 backbone
        self.backbone = resnet50()
        # self.backbone = ResNet50()
        del self.backbone.fc
        self.conv_ch = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1)
        # create conversion layer
        self.conv = nn.Conv2d(2048, hidden_dim, 1)
        self.hidden_dim = hidden_dim
        # create a default PyTorch transformer
        self.transformer = nn.Transformer(
            hidden_dim, nheads, num_encoder_layers, num_decoder_layers, batch_first=False)
        self.embed_dim = hidden_dim * 100
        # prediction heads, one extra class for predicting non-empty slots
        # note that in baseline DETR linear_bbox layer is 3-layer MLP
        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
        self.linear_bbox = nn.Linear(hidden_dim, 4)
        self.linear_center = nn.Linear(hidden_dim, 2)
        self.linear_height_width = nn.Linear(hidden_dim, 2)
        self.linear_yaw = nn.Linear(hidden_dim, 1)
        self.linear_embed = nn.Linear(self.embed_dim, hidden_dim)
        # output positional encodings (object queries)
        self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))

        # spatial positional encodings
        # note that in baseline DETR we use sine positional encodings
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

    def forward(self, inputs):
        # propagate inputs through ResNet-50 up to avg-pool layer
        # x = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1)(inputs)
        x = self.conv_ch(inputs)
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        # convert from 2048 to 256 feature planes for the transformer
        h = self.conv(x)
        # construct positional encodings
        H, W = h.shape[-2:] # height and width of the bounding box
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)
        # propagate through the transformer
        h = self.transformer(pos + 0.1 * h.flatten(2).permute(2, 0, 1),
                             self.query_pos.unsqueeze(1)).transpose(0, 1)
        shape_list = h.shape
        # h = torch.reshape(h, (shape_list[0], shape_list[1] * shape_list[2]))
        h = torch.reshape(h, (1, self.embed_dim))
        # self.embed_dim = shape_list[1] * shape_list[2]
        # h = nn.Linear(shape_list[1] * shape_list[2], self.hidden_dim)(h)
        h = self.linear_embed(h)
        return self.linear_center(h).sigmoid(), self.linear_yaw(h).sigmoid(), self.linear_height_width(h).sigmoid()

class StarDataset(torch.utils.data.Dataset):
    """Return star image and labels"""

    def __init__(self, data_size=500):
        self.data_size = data_size

    def __len__(self) -> int:
        return self.data_size

    def __getitem__(self, idx) -> t.Tuple[torch.Tensor, torch.Tensor]:
        image, label = synthesize_data(has_star=True)
        return image[None], label

  
def train(loss_fn_para: [1, 10, 1], model: DETRdemo, dl: StarDataset, num_epochs: int, image_width: int, image_height: int) -> DETRdemo:

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(num_epochs):
        print(f"EPOCH: {epoch}")
        losses = []
        for image, label in tqdm(dl, total=len(dl)):
            image = image.to(DEVICE).float()
            label = label.to(DEVICE).float()
            # print(image.shape)
            optimizer.zero_grad()
            preds = model(image)       
            # print(torch.stack([preds[0][0][0] * image_width, preds[0][0][1] * image_height], dim=-1))
            # print(preds[1][0] * 2 * math.pi)
            # print(torch.stack([preds[2][0][0] * image_width, preds[2][0][1] * image_height], dim=-1))
            loss = loss_fn(torch.stack([preds[0][0][0] * image_width, preds[0][0][1] * image_height], dim=-1), label[0][:2]) * loss_fn_para[0] + \
                   loss_fn(preds[1][0] * 2 * math.pi, label[0][2]) * loss_fn_para[1] + \
                   loss_fn(torch.stack([preds[2][0][0] * image_width, preds[2][0][1] * image_height], dim=-1), label[0][3:5]) * loss_fn_para[2]
            loss.backward()
            losses.append(loss.detach().cpu().numpy())
            optimizer.step()
        print(np.mean(losses))

    return model


def main():
    batch_size = 1
    data_size = 5000
    image_width = 200
    image_height = 200
    print("fff1")
    model = DETRdemo(num_classes=2).to(DEVICE)
    # summary(model, (1, image_size, image_size), batch_size=1)
    loss_fn_para = [1, 10, 1]
    data_set = StarDataset(data_size=data_size)
    star_model = train(
        loss_fn_para,
        model,
        torch.utils.data.DataLoader(data_set, batch_size=batch_size, num_workers=8),
        num_epochs=20,
        image_width=image_width,
        image_height=image_height
    )
    torch.save(star_model.state_dict(), "model.pickle")


if __name__ == "__main__":
    main()
