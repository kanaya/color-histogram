# 色相ヒストグラム

## 環境のインストール方法

```
$ python3 -m venv env
$ . ./env/bin/activate
$ pip3 install numpy
$ pip3 install opencv-python
$ pip3 install matplotlib
```

## 基本的な実行方法

```
$ python3 color-histogram {画像ファイル1}.jpg
```

本プログラムは`{画像ファイル1}.csv`を出力する．

## オプション

```
-h
```

ヘルプの表示

```
--show-histogram
```

ヒストグラムを表示する．

```
--threshold {TH}
```

彩度が`{TH}`未満の場合は色相ヒストグラム（おかず）の対象としない．

```
--threshold-rice {TH_RICE}
```

明度が`{TH_RICE}`未満の場合は白色ヒストグラム（ごはん）の対象としない．

```
-r-lower {R_LOWER}
```

赤の範囲の下限を`{R_LOWER}`にする．

```
-r-upper {R_UPPER}
```

赤の範囲の上限を`{R_UPPER}`にする．


## 実行方法の例

### 複数の画像ファイルから色相ヒストグラムを生成する

```
python3 color-histogram.py *.jpg
```

### 色相ヒストグラムを視覚的に確認する

```
python3 color-histogram.py --show-histogram {画像ファイル1}.jpg
```

## 終了方法

```
$ deactivate
```