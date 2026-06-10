# FC Tokyo Attendance Analysis

FC東京ホームゲームの入場者数を分析・予測するための notebook と共通コードをまとめたリポジトリです。

## Structure

```text
.
├── data/
│   ├── raw/                 # 元データ
│   └── processed/           # 加工済みデータ
├── notebooks/
│   ├── 01_data_collection/  # データ取得・抽出
│   ├── 02_eda/              # 探索的データ分析
│   ├── 03_feature_engineering/
│   ├── 04_modeling/         # 予測モデル
│   └── 05_explainability/   # モデル解釈
├── fctokyo_modeling.py      # 共通の前処理・評価関数
└── requirements.txt
```

## Run

```bash
.venv/bin/python fctokyo_modeling.py
```
