import typing as t

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from utils import DEVICE, synthesize_data
from torchsummary import summary
### triplet loss pytorch for yaw
import torch
from torch import nn
torch.set_grad_enabled(False)
from torchvision.models import resnet50

class StarModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv0 = nn.Conv2d(1, 32, 3)
        self.conv1 = nn.Conv2d(32, 64, 3)
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.conv3 = nn.Conv2d(128, 256, 3)
        self.conv4 = nn.Conv2d(256, 512, 3)
        self.conv5 = nn.Conv2d(512, 512, 3)

        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv0(x)))
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class DETRdemo(nn.Module):
    """
    Demo DETR implementation.

    Demo implementation of DETR in minimal number of lines, with the
    following differences wrt DETR in the paper:
    * learned positional encoding (instead of sine)
    * positional encoding is passed at input (instead of attention)
    * fc bbox predictor (instead of MLP)
    The model achieves ~40 AP on COCO val5k and runs at ~28 FPS on Tesla V100.
    Only batch size 1 supported.
    """

    def __init__(self, num_classes=2, hidden_dim=256, nheads=8,
                 num_encoder_layers=6, num_decoder_layers=6):
        super().__init__()

        # create ResNet-50 backbone
        self.backbone = resnet50()
        # self.backbone = ResNet50()
        del self.backbone.fc

        # create conversion layer
        self.conv = nn.Conv2d(2048, hidden_dim, 1)

        # create a default PyTorch transformer
        self.transformer = nn.Transformer(
            hidden_dim, nheads, num_encoder_layers, num_decoder_layers)

        # prediction heads, one extra class for predicting non-empty slots
        # note that in baseline DETR linear_bbox layer is 3-layer MLP
        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
        self.linear_bbox = nn.Linear(hidden_dim, 4)
        self.linear_center = nn.Linear(hidden_dim, 2)
        self.linear_height_width = nn.Linear(hidden_dim, 2)
        self.linear_yaw = nn.Linear(hidden_dim, 1)

        # output positional encodings (object queries)
        self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))

        # spatial positional encodings
        # note that in baseline DETR we use sine positional encodings
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

    def forward(self, inputs):
        # propagate inputs through ResNet-50 up to avg-pool layer
        x = self.backbone.conv1(inputs)
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

        return self.linear_center(h).relu(), self.linear_yaw(h).relu(), self.linear_height_width(h).relu()

        # # finally project transformer outputs to class labels and bounding boxes
        # return {'pred_logits': self.linear_class(h),
        #         'pred_boxes': self.linear_bbox(h).sigmoid()}

class StarDataset(torch.utils.data.Dataset):
    """Return star image and labels"""

    def __init__(self, data_size=500):
        self.data_size = data_size

    def __len__(self) -> int:
        return self.data_size

    def __getitem__(self, idx) -> t.Tuple[torch.Tensor, torch.Tensor]:
        image, label = synthesize_data(has_star=True)
        return image[None], label


def train(loss_fn_para: [1, 10, 1], model: StarModel, dl: StarDataset, num_epochs: int) -> StarModel:

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(num_epochs):
        print(f"EPOCH: {epoch}")
        losses = []
        for image, label in tqdm(dl, total=len(dl)):
            image = image.to(DEVICE).float()
            label = label.to(DEVICE).float()

            optimizer.zero_grad()

            preds = model(image)
            # loss = loss_fn(preds, label)
            loss = loss_fn(preds[:2], label[:2]) * loss_fn_para[0] + \
                   loss_fn(preds[2], label[2]) * loss_fn_para[1] + \
                   loss_fn(preds[3:5], label[3:5]) * loss_fn_para[2]
            loss.backward()
            losses.append(loss.detach().cpu().numpy())
            optimizer.step()
        print(np.mean(losses))

    return model


def main():
    model = StarModel().to(DEVICE)
    # model = DETRdemo().to(DEVICE)
    loss_fn_para = [1, 10, 1]
    star_model = train(
        loss_fn_para,
        model,
        torch.utils.data.DataLoader(StarDataset(), batch_size=1, num_workers=8),
        num_epochs=30,
    )
    torch.save(star_model.state_dict(), "model.pickle")


if __name__ == "__main__":
    main()
