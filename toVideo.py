import os
from cv2 import VideoWriter, VideoWriter_fourcc, imread

dataDir = "CamVid/701_StillsRaw_full"

dataDir = 'result/resultSequence'
dataList = os.listdir(dataDir)
imgList = [os.path.join(dataDir, file) for file in dataList if file.find('001TP') != -1]
imgList.sort(key=lambda x: int(x[-10:-5]))


def write(images, outimg=None, fps=5, size=None, is_color=True, format="XVID", outvid='demo.avi'):
    fourcc = VideoWriter_fourcc(*format)
    vid = VideoWriter(outvid, fourcc, float(fps), size, is_color)
    for image in images:
        img = imread(image)
        vid.write(img)
    vid.release()
    return vid


write(imgList, fps=5, size=(960, 720), outvid='result/video/result.mp4', format='mp4v')
