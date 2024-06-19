# Image Segmentation for Detection of Damaged Part of Leaf and detecting its Intensity

import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt

files = glob.glob('D:/Victor/tomato_leaf_detection/train/Tomato___Bacterial_spot/*.jpg')
    
# Scaleogram Image Generation of Images
    
import matplotlib.pyplot as plt
import pywt
from scipy import ndimage
import io
from PIL import Image


def fig2img(fig):
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img   

dir_name = 'D:/Sintu/scaleogram_img'
i = 0

for i in range(len(files)):
    img = plt.imread(files[i])
    img_gray = np.mean(img, axis=2)
    coeffs = pywt.dwt2(img_gray, 'haar')
    scaleogram = np.abs(coeffs[0])
    plt.imshow(scaleogram, cmap='coolwarm')
    fig = plt.gcf()
    img = fig2img(fig)
    img.save(dir_name + '/fig_' + str(i) + '.png')