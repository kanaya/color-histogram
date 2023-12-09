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

R_LOWER_DEFAULT = 0
R_UPPER_DEFAULT = 5

BR_LOWER_DEFAULT = 2
BR_UPPER_DEFAULT = 8

Y_LOWER_DEFAULT = 2
Y_UPPER_DEFAULT = 11

G_LOWER_DEFAULT = 8
G_UPPER_DEFAULT = 24

B_LOWER_DEFAULT = 30
B_UPPER_DEFAULT = 35

parser = argparse.ArgumentParser(description='Create histogram of hue from images.')
parser.add_argument('image', metavar='IMG', nargs='+', help='image file')

parser.add_argument('--threshold', dest='threshold', metavar='TH', choices=range(0, 256), action='store', default=TH_DEFAULT, help='threshold (default: {})'.format(TH_DEFAULT))
parser.add_argument('--n-bins', dest='n_bins', metavar='N_BINS', choices=range(1, 72), action='store', default=N_BINS_DEFAULT, help='number of bins (default: {})'.format(N_BINS_DEFAULT))

parser.add_argument('--r-lower', dest='r_lower_limit', metavar='R_LOWER', choices=range(0, 72), action='store', default=R_LOWER_DEFAULT, help='R, lower limit (default: {})'.format(R_LOWER_DEFAULT))
parser.add_argument('--r-upper', dest='r_upper_limit', metavar='R_UPPER', choices=range(0, 72), action='store', default=R_UPPER_DEFAULT, help='R, upper limit (default: {})'.format(R_UPPER_DEFAULT))

parser.add_argument('--br-lower', dest='br_lower_limit', metavar='BR_LOWER', choices=range(0, 72), action='store', default=BR_LOWER_DEFAULT, help='BR, lower limit (default: {})'.format(BR_LOWER_DEFAULT))
parser.add_argument('--br-upper', dest='br_upper_limit', metavar='BR_UPPER', choices=range(0, 72), action='store', default=BR_UPPER_DEFAULT, help='BR, upper limit (default: {})'.format(BR_UPPER_DEFAULT))

parser.add_argument('--y-lower', dest='y_lower_limit', metavar='Y_LOWER', choices=range(0, 72), action='store', default=Y_LOWER_DEFAULT, help='Y, lower limit (default: {})'.format(Y_LOWER_DEFAULT))
parser.add_argument('--y-upper', dest='y_upper_limit', metavar='Y_UPPER', choices=range(0, 72), action='store', default=Y_UPPER_DEFAULT, help='Y, upper limit (default: {})'.format(Y_UPPER_DEFAULT))

parser.add_argument('--g-lower', dest='g_lower_limit', metavar='G_LOWER', choices=range(0, 72), action='store', default=G_LOWER_DEFAULT, help='G, lower limit (default: {})'.format(G_LOWER_DEFAULT))
parser.add_argument('--g-upper', dest='g_upper_limit', metavar='G_UPPER', choices=range(0, 72), action='store', default=G_UPPER_DEFAULT, help='G, upper limit (default: {})'.format(G_UPPER_DEFAULT))

parser.add_argument('--b-lower', dest='b_lower_limit', metavar='B_LOWER', choices=range(0, 72), action='store', default=B_LOWER_DEFAULT, help='B, lower limit (default: {})'.format(B_LOWER_DEFAULT))
parser.add_argument('--b-upper', dest='b_upper_limit', metavar='B_UPPER', choices=range(0, 72), action='store', default=B_UPPER_DEFAULT, help='B, upper limit (default: {})'.format(B_UPPER_DEFAULT))

parser.add_argument('--show-histogram', dest='show_histogram', action='store_const', default=False, const=True, help='show histogram')
parser.add_argument('--show-mask', dest='show_mask', action='store_const', default=False, const=True,help='show mask image')
args = parser.parse_args()

th = args.threshold
r_lower = args.r_lower_limit

for img_name in args.image:
	img = cv2.imread(img_name)
	img_height, img_width = img.shape[:2]
	img_size = img_height * img_width
	img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	img_s = img_hsv[..., 1]
	ret, img_s_th = cv2.threshold(img_s, th, 255, cv2.THRESH_BINARY)
	n_pixel_img_s_th = np.sum(img_s_th) / 255
	if args.show_mask:
		cv2.imshow("saturation", img_s_th)
	hue_histogram = cv2.calcHist([img_hsv], channels=[0], mask=img_s_th, histSize=[args.n_bins], ranges=[0, 180])
	hue_histogram_v = np.ravel(hue_histogram)
	hue_histogram_v /= n_pixel_img_s_th
	img_v = img_hsv[..., 2]
	value_histogram = cv2.calcHist([img_hsv], channels=[2], mask=None, histSize=[4], ranges=[0, 256])
	value_histogram_v = np.ravel(value_histogram)
	value_histogram_v /= img_size
	the_histogram = np.append(hue_histogram_v, value_histogram_v[3])
	csv_name = Path(img_name).stem + '.csv'
	csv = open(csv_name, "w")
	for i in range(args.n_bins + 1):
		print("{}, {}".format(i, the_histogram[i]), file=csv)
	csv.close()
	if args.show_histogram:
		color_list = [colorsys.hsv_to_rgb(h / args.n_bins, 1.0, 1.0) for h in range(args.n_bins)]
		color_list.append([0.75, 0.75, 0.75])
		fig, ax = plt.subplots()
		ax.bar(np.array([i for i in range(args.n_bins + 1)]), the_histogram, color=color_list)
		plt.show()
