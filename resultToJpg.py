import os
import numpy as np
import scipy.misc
import torch
from PIL import Image


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
    print(outputNumpy.shape)
    return outputNumpy

modelPath = 'models/Deeplabv3_epoch:499_accu:0.9155845114087301'
rootDir = "CamVid/"
dataDir = os.path.join(rootDir, "701_StillsRaw_full")

label_colors_file = os.path.join(rootDir, "label_colors.txt")
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

dataList = os.listdir(dataDir)
imgList = [os.path.join(dataDir, file) for file in dataList if file.find('001TP') != -1]
imgList.sort(key=lambda x: int(x[-10:-5]))

saveList = [os.path.join('./resultSequence', file) for file in dataList if file.find('001TP') != -1]
saveList.sort(key=lambda x: int(x[-10:-5]))

means = np.array([103.939, 116.779, 123.68]) / 255.

model = torch.load(modelPath)

for i, imgFile in enumerate(imgList):
    img = scipy.misc.imread(imgFile, mode='RGB')
    img = img[:, :, ::-1]  # switch to BGR
    img = np.transpose(img, (2, 0, 1)) / 255.
    img[0] -= means[0]
    img[1] -= means[1]
    img[2] -= means[2]

    img = torch.from_numpy(img.copy()).float().cuda().unsqueeze(0)

    output = model(img)
    output = output[0].argmax(0).byte().cpu().numpy()

    imgSave = Image.fromarray(np.uint8(transfer(output))).convert('RGB')
    imgSave.save(saveList[i])
