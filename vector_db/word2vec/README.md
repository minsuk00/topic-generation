## word2vecの使い方

### 1. データの準備
from_s3に必要なデータをimportしてくる
サンプルデータとして a.json, b.json, c.json を用意しているが本番環境ではs3からデータを取得する

### 2. transform.py
title + abstract -> 756次元のベクトルに変換する
変換したデータは
```json
[{'id':XXX, 'vector':[0.1, 0.2, ...]}, ...]
```
の形式で vectors/ 以下にgzip形式で保存される

注意点として、以下の変数を適宜変更する必要がある。
TODOとして環境変数としたい。
```python
TOTAL_WORKS_COUNT = 35896867 # Worksの数。TODOとして自動で取得するようにする
CHUNK_SIZE = 100000 # 1回の処理で変換するWorkの数. この単位でgzipに保存される。目安としてg5 instanceで10分程度かかりgzip後に700MB程度のファイルができる
START_PAGE = 15 # プロセスが落ちたときに途中から再開するための変数。この値を変更することで途中から再開できる
START_LINE = START_PAGE * CHUNK_SIZE 
```

## 実行環境

### Platform
- Ubuntu

### AMI name
- Deep Learning AMI GPU PyTorch 2.1.0 (Ubuntu 20.04) 20231103

### Instance type
- g5.2xlarge
- 目安として、$1.212/hour