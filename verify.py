from ultralytics import YOLO

model = YOLO(
    "ultralytics/cfg/models/26/yolo26n-pan.yaml",
    task="segment",
)

model.info()

import torch

net = model.model
net.train()

x = torch.randn(2, 3, 640, 640)

with torch.no_grad():
    pred = net(x)

print(pred.keys())

print(pred["one2many"]["proto"].shape)
print(pred["one2many"]["stuff_logits"].shape)
print(pred["one2many"]["aux_stuff_logits"].shape)
print(pred["one2many"]["mask_coefficient"].shape)