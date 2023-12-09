import sys
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import colorsys
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pathlib import Path

N_BINS_DEFAULT = 72
TH_DEFAULT = 50

parser = argparse.ArgumentParser(description='Show range of hue from images.')
parser.add_argument('image', metavar='IMG', nargs='+', help='image file')
parser.add_argument('--threshold', dest='threshold', metavar='TH', choices=range(0, 256), action='store', default=TH_DEFAULT, help='threshold (default: {})'.format(TH_DEFAULT))
parser.add_argument('--n-bins', dest='n_bins', metavar='N_BINS', choices=range(1, 72), action='store', default=N_BINS_DEFAULT, help='number of bins (default: {})'.format(N_BINS_DEFAULT))
parser.add_argument('--show-histogram', dest='show_histogram', action='store_const', default=False, const=True, help='show histogram')
args = parser.parse_args()

th = args.threshold

for img_name in args.image:
	img = cv2.imread(img_name)
	img_height, img_width = img.shape[:2]
	img_size = img_height * img_width
	img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	img_s = img_hsv[..., 1]
	ret, img_s_th = cv2.threshold(img_s, th, 255, cv2.THRESH_BINARY)
	n_pixel_img_s_th = np.sum(img_s_th) / 255
	hue_histogram = cv2.calcHist([img_hsv], channels=[0], mask=img_s_th, histSize=[args.n_bins], ranges=[0, 180])
	hue_histogram_v = np.ravel(hue_histogram)
	hue_histogram_v /= n_pixel_img_s_th
	hue_histogram_list = hue_histogram_v.tolist()
	lower_limit = min((i for i in range(len(hue_histogram_list)) if hue_histogram_list[i] >= 0.05), default = -1)
	upper_limit = max((i for i in range(len(hue_histogram_list)) if hue_histogram_list[i] >= 0.05), default = -1)
	print("lower = {}, upper = {}".format(lower_limit, upper_limit))