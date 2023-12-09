import sys
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import colorsys
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pathlib import Path

# 色相ヒストグラムのデフォルト階数
N_BINS_DEFAULT = 72

# 彩度によるマスクのしきい値（0-255），おかず部分の認識用
TH_DEFAULT = 50

# 明度によるマスクのしきい値（0-255），ご飯部分の認識用
TH_RICE_DEFAULT = 150

# R, BR, Y, G, B 色相の範囲（72段階換算）
N_COLORS = 5
COLORS = ['R', 'BR', 'Y', 'G', 'B']

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

# 引数を解析する
parser = argparse.ArgumentParser(description='Create histogram of hue from images.')
parser.add_argument('image', metavar='IMG', nargs='+', help='image file')

parser.add_argument('--threshold', dest='threshold', metavar='TH', choices=range(0, 256), action='store', default=TH_DEFAULT, help='saturation threshold for okazu (default: {})'.format(TH_DEFAULT))
parser.add_argument('--threshold-rice', dest='threshold_rice', metavar='TH_RICE', choices=range(0, 256), action='store', default=TH_RICE_DEFAULT, help='value threshold for rice (default: {})'.format(TH_RICE_DEFAULT))

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

args = parser.parse_args()

# 大域変数を初期化する
th = args.threshold
th_rice = args.threshold_rice
r_lower = args.r_lower_limit
r_upper = args.r_upper_limit
br_lower = args.br_lower_limit
br_upper = args.br_upper_limit
y_lower = args.y_lower_limit
y_upper = args.y_upper_limit
g_lower = args.g_lower_limit
g_upper = args.g_upper_limit
b_lower = args.b_lower_limit
b_upper = args.b_upper_limit

# 本体
for img_name in args.image:
	# RGB画像を読み込む
	img = cv2.imread(img_name)
	img_height, img_width = img.shape[:2]
	img_size = img_height * img_width
	# RGB画像をHSV画像へ変換する
	img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	# HSV画像からS画像を抜き出す
	img_s = img_hsv[..., 1]
	# S画像をマスクのしきい値（th）で2値化してSマスク画像を作る
	ret, img_okazu = cv2.threshold(img_s, th, 255, cv2.THRESH_BINARY)
	# 2値化したS画像の白ピクセル数を数える（白ピクセル値が255なので全体を255で割る）
	n_pixel_img_okazu = np.sum(img_okazu) / 255
	# HSV画像から色相のヒストグラムを生成する
	hue_histogram = cv2.calcHist([img_hsv], channels=[0], mask=img_okazu, histSize=[args.n_bins], ranges=[0, 180])
	# 配列を平坦化する
	hue_histogram_flat = np.ravel(hue_histogram)
	# HSV画像からV画像を抜き出す
	img_v = img_hsv[..., 2]
	# V画像からご飯のしきい値(th_rice)で2値化する
	ret, img_v_th = cv2.threshold(img_v, th_rice, 255, cv2.THRESH_BINARY)
	# Sマスク画像の反転画像を作る
	img_okazu_negative = cv2.bitwise_not(img_okazu)
	# img_v_thとimg_okazu_negativeの積をとる
	img_gohan = cv2.bitwise_and(img_v_th, img_okazu_negative)
	# ご飯画像の白ピクセル数を数える
	n_pixel_img_gohan = np.sum(img_gohan) / 255
	# おかず部分とご飯部分の合計を計算する
	n_pixels = n_pixel_img_okazu + n_pixel_img_gohan
	# おかず色相ヒストグラムを全体のパーセンテージに変換する
	hue_histogram_flat /= n_pixels
	# ご飯部分のピクセル数を全体のパーセンテージに変換する
	gohan = n_pixel_img_gohan / n_pixels
	## デバッグ用
	## cv2.imshow("gohan", img_gohan)
	## cv2.waitKey(0)
	# おかず色相ヒストグラムにご飯部分を付け足す
	the_histogram = np.append(hue_histogram_flat, [gohan])
	# 全体を100倍する
	the_histogram *= 100
	# CSVファイルに保存する
	csv_name = Path(img_name).stem + '.csv'
	csv = open(csv_name, "w")
	for i in range(args.n_bins + 1):
		print("{}, {}".format(i, the_histogram[i]), file=csv)
	csv.close()
	# 心理的ヒストグラムを作成する（色相は72段階であることを前提とする）
	lower_limits = [r_lower, br_lower, y_lower, g_lower, b_lower]
	upper_limits = [r_upper, br_upper, y_upper, g_upper, b_upper]
	uneven_histogram = []
	for c in range(0, N_COLORS):
		lower_limit = lower_limits[c]
		upper_limit = upper_limits[c]
		slice = the_histogram[lower_limit:upper_limit]
		s = sum(slice)
		uneven_histogram.append(s)
		## print('{}: {}%'.format(COLORS[c], s))
	# CSVファイルに保存する
	csv_name2 = Path(img_name).stem + '-uneven.csv'
	csv2 = open(csv_name2, "w")
	for i in range(N_COLORS):
		print("{}, {}".format(COLORS[i], uneven_histogram[i]), file=csv2)
	csv2.close()
	# --show-histogramオプションが指定されていたら，ヒストグラムを画面表示する
	if args.show_histogram:
		color_list = [colorsys.hsv_to_rgb(h / args.n_bins, 1.0, 1.0) for h in range(args.n_bins)]
		color_list.append([0.75, 0.75, 0.75])
		fig, ax = plt.subplots()
		ax.bar(np.array([i for i in range(args.n_bins + 1)]), the_histogram, color=color_list)
		plt.show()
