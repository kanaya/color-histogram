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

parser = argparse.ArgumentParser(description='Create histogram of hue from images.')
parser.add_argument('image', metavar='IMG', nargs='+', help='image file')
parser.add_argument('--threshold', dest='threshold', metavar='TH', choices=range(0, 256),
					action='store', default=TH_DEFAULT,
					help='threshold (default: {})'.format(TH_DEFAULT))
parser.add_argument('--n-bins', dest='n_bins', metavar='N_BINS', choices=range(1, 72),
					action='store', default=N_BINS_DEFAULT,
					help='number of bins (default: {})'.format(N_BINS_DEFAULT))
parser.add_argument('--show-histogram', dest='show_histogram',
					action='store_const', default=False, const=True, help='show histogram')
parser.add_argument('--show-mask', dest='show_mask', action='store_const', default=False, const=True,
					help='show mask image')
args = parser.parse_args()

th = args.threshold

for img_name in args.image:
	img = cv2.imread(img_name)
	img_height, img_width = img.shape[:2]
	img_size = img_height * img_width
	img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	img_s = img_hsv[..., 1]
	ret, img_s_th = cv2.threshold(img_s, th, 255, cv2.THRESH_BINARY)
	if args.show_mask:
		cv2.imshow("saturation", img_s_th)
	hue_histogram = cv2.calcHist([img_hsv], channels=[0], mask=img_s_th, histSize=[args.n_bins], ranges=[0, 180])
	hue_histogram_v = np.ravel(hue_histogram)
	hue_histogram_v /= img_size
	img_v = img_hsv[..., 2]
	value_histogram = cv2.calcHist([img_hsv], channels=[2], mask=None, histSize=[4], ranges=[0, 256])
	value_histogram_v = np.ravel(value_histogram)
	value_histogram_v /= img_size
	the_histogram = np.append(hue_histogram_v, value_histogram_v[3])
	csv_name = Path(img_name).stem + '.csv'
	csv = open(csv_name, "w")
	for i in range(args.n_bins):
		print("{}, {}".format(i, the_histogram[i]), file=csv)
	csv.close()
	if args.show_histogram:
		color_list = [colorsys.hsv_to_rgb(h / args.n_bins, 1.0, 1.0) for h in range(args.n_bins)]
		color_list.append([0.75, 0.75, 0.75])
		fig, ax = plt.subplots()
		ax.bar(np.array([i for i in range(args.n_bins + 1)]), the_histogram, color=color_list)
		plt.show()
