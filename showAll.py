import torch
from CamVid_loader import CamVidDataset
import os
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np

root_dir = "CamVid/"
val_file = os.path.join(root_dir, "val.csv")
modelPath = 'models/Deeplabv3_epoch:499_accu:0.9155845114087301'

val_data = CamVidDataset(csv_file=val_file, phase='val', flip_rate=8)
model = torch.load(modelPath)

label_colors_file = os.path.join(root_dir, "label_colors.txt")
label2color = {}
color2label = {}
label2index = {}
index2label = {}

f = open(label_colors_file, "r").read().split("\n")[:-1]  # ignore the last empty line
for idx, line in enumerate(f):
    label = line.split()[-1]
    color = tuple([int(x) for x in line.split()[:-1]])
    label2color[label] = color
    color2label[color] = label
    label2index[label] = idx
    index2label[idx] = label


def transfer(inputNumpy):
    outputNumpy = np.zeros((inputNumpy.shape[0], 720, 960, 3))
    for i in range(inputNumpy.shape[0]):
        for h in range(720):
            for w in range(920):
                index = inputNumpy[i, h, w]
                try:
                    label = index2label[index]
                    color = label2color[label]
                    outputNumpy[i, h, w] = color
                except:
                    print("error: h:%d, w:%d" % (h, w))
    return outputNumpy


n_class = 32

stride = 15

inputs = torch.stack([val_data[i]['X'].cuda() for i in range(0, 70, stride)], 0)
origins = torch.stack([val_data[i]['l'] for i in range(0, 70, stride)], 0).byte().cpu().numpy()

outputs = model(inputs)
outputs = outputs.argmax(1).byte().cpu().numpy()

outputs = transfer(outputs)
origins = transfer(origins)

outputs = np.concatenate([outputs[i] for i in range(outputs.shape[0])], axis=1)
origins = np.concatenate([origins[i] for i in range(origins.shape[0])], axis=1)

img1 = Image.fromarray(np.uint8(outputs)).convert('RGB')
img2 = Image.fromarray(np.uint8(origins)).convert('RGB')

width = img1.size[0]
height = img1.size[1]
dpi = 10
plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
plt.subplot(2, 1, 1)
plt.imshow(img1)
plt.axis('off')
plt.subplot(2, 1, 2)
plt.imshow(img2)
plt.axis('off')
plt.savefig('result/showAll/output.jpg')
plt.show()
