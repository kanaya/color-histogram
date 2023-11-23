import sys
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import colorsys
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pathlib import Path

N_BINS = 72
TH_DEFAULT = 50

parser = argparse.ArgumentParser(description='Create histogram of hue from images.')
parser.add_argument('image', metavar='IMG', nargs='+', help='image file')
parser.add_argument('--th', dest='threshold', metavar='TH', choices=range(0, 255), default=TH_DEFAULT,
					 help='threshold (default: {})'.format(TH_DEFAULT))
args = parser.parse_args()

img_name = args.image[0]
th = args.threshold

img = cv2.imread(img_name)
img_height, img_width = img.shape[:2]
img_size = img_height * img_width
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
img_s = img_hsv[..., 1]
ret, img_s_th = cv2.threshold(img_s, th, 255, cv2.THRESH_BINARY)
cv2.imshow("saturation", img_s_th)

hue_histogram = cv2.calcHist([img_hsv], [0], img_s_th, [N_BINS], [0, 180])
hue_histogram_v = np.ravel(hue_histogram)
hue_histogram_v /= img_size

csv_name = Path(img_name).stem + '.csv'
csv = open(csv_name, "w")
for i in range(N_BINS):
	print("{}, {}".format(i, hue_histogram_v[i]), file=csv)
csv.close()

color_list = [colorsys.hsv_to_rgb(h / N_BINS, 1.0, 1.0) for h in range(N_BINS)]
fig, ax = plt.subplots()
ax.bar(np.array([i for i in range(N_BINS)]), hue_histogram_v, color=color_list)
plt.show()
