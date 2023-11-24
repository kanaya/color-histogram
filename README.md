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
--threshold {TH}
```

彩度が`{TH}`未満の場合は色相ヒストグラムの対象としない．

```
--n-bins {N_BINS}
```

色相ヒストグラムの階層を`{N_BINS}`にする．

```
--show-histogram
```

ヒストグラムを表示する．

```
--show-mask
```

マスク画像を表示する．（`--show-histogram`と同時に指定すること．）

## 実行方法の例

### 複数の画像ファイルから色相ヒストグラムを生成する

```
$ python3 color-histogram.py *.jpg
```

### 色相ヒストグラムを視覚的に確認する

```
$ python3 color-histogram.py --show-histogram {画像ファイル1}.jpg
```

### 色相ヒストグラムとマスク画像を視覚的に確認する

```
$ python3 color-histogram.py --show-histogram --show-mask {画像ファイル1}.jpg
```

## 終了方法

```
$ deactivate
```