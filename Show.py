import torch
from CamVid_loader import CamVidDataset
import os
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np

def transfer(inputNumpy):
    outputNumpy = np.zeros((720, 960, 3))
    for h in range(720):
        for w in range(920):
            index = inputNumpy[h, w]
            try:
                label = index2label[index]
                color = label2color[label]
                outputNumpy[h, w] = color
            except:
                print("error: h:%d, w:%d" % (h, w))
    return outputNumpy


n_class = 32

modelPath = 'models/Deeplabv3_epoch:499_accu:0.9155845114087301'
root_dir = "CamVid/"
val_file = os.path.join(root_dir, "val.csv")

val_data = CamVidDataset(csv_file=val_file, phase='val', flip_rate=8)
model = torch.load(modelPath)

value = val_data[0]

inputs = value['X'].cuda().unsqueeze(0)
output = model(inputs)
output = output[0].argmax(0).byte().cpu().numpy()
origin = value['l'].byte().cpu().numpy()

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

img1 = Image.fromarray(np.uint8(transfer(origin))).convert('RGB')
img2 = Image.fromarray(np.uint8(transfer(output))).convert('RGB')

plt.imshow(img1)
plt.axis('off')
plt.savefig('result/show/origin.jpg')
plt.show()
plt.imshow(img2)
plt.axis('off')
plt.savefig('result/show/output.jpg')
plt.show()
