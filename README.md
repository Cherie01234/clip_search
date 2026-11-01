# 🔍 ビジュアル類似性検索（Visual Similarity Search）

[![CI](https://github.com/Cherie01234/clip_search/actions/workflows/ci.yml/badge.svg)](https://github.com/Cherie01234/clip_search/actions/workflows/ci.yml)

## 概要

| 項目 | 内容 |
|---|---|
| 解く課題 | タグや説明文のない大量の画像から、キーワード（日本語・英語）や手持ちの画像に意味が近いものを探す |
| 手法 | 多言語CLIP（open-clip-torch の `xlm-roberta-base-ViT-B-32`）で画像とテキストを同じ512次元のベクトル空間に埋め込み、L2正規化したベクトルを FAISS（内積＝コサイン類似度）で近傍検索。Streamlitで検索画面を提供 |
| 評価 | 約4.4万枚（44,441枚）の画像でインデックスを構築して検索（下記デモ）。検索精度の定量評価（正解ラベルを使った Recall@K など）は未実施。テストはFAISSによる近傍の順位付けと、インデックスの保存・読み込みを対象に整備（pytest） |
| 使い方 | 同梱のサンプル画像で試す場合：`python make_sample_images.py` → `python build_index.py --image-dir sample_images --index-dir index` → `streamlit run app.py` |
| 開発 | 設計から個人で開発（開発にはAIエージェントを活用）。pytest と GitHub Actions の CI を整備 |

---

多言語 CLIP × FAISS によるテキスト・画像検索アプリ。
キーワード（**日本語・英語**）や手持ちの画像から、画像コレクションの中で意味的に近いものを検索できます。

CLIP は画像とテキストを同じベクトル空間に埋め込むため、テキストだけで画像をゼロショット検索できます。

## デモ

「red dress」「running shoes」のようにキーワードを入力すると、約4.4万枚の中から似た画像を即座に提示します。

![デモ](docs/demo.gif)

## 仕組み

![アーキテクチャ](docs/architecture.svg)

## 構成

| ファイル | 役割 |
|----------|------|
| `app.py` | Streamlit 製の検索 UI |
| `encoder.py` | CLIP による画像/テキストのベクトル化 |
| `indexer.py` | FAISS（内積＝コサイン類似度）による近傍検索 |
| `build_index.py` | 画像フォルダからインデックスを構築するスクリプト |
| `app_start.bat` | Windows 用のワンクリック起動バッチ |

> 注: 画像データ (`data/`) と生成済みインデックス (`index/`) はリポジトリに含めていません。
> 各自で画像を用意し、`build_index.py` でインデックスを構築してください。

## セットアップ

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

## ⚡ クイックお試し（サンプル画像同梱）

画像を用意しなくても、同梱のサンプル（簡単なアイコン10枚）ですぐ動作確認できます:

```bash
python make_sample_images.py                                  # sample_images/ を生成
python build_index.py --image-dir sample_images --index-dir index
streamlit run app.py                                          # 「星」「house」「魚」などで検索
```

> サンプルは動作確認用の簡易アイコンです。本来の意味検索の威力は実写真コレクションで発揮されます。

## 使い方

### 1. インデックスを構築（初回のみ）

検索対象の画像を `data/images/` に置き、以下を実行します。

```bash
python build_index.py --image-dir data/images --index-dir index
```

対応形式: `.jpg .jpeg .png .webp .bmp`

### 2. アプリを起動

```bash
streamlit run app.py
```

Windows では `app_start.bat` をダブルクリックでも起動できます。
ブラウザで `http://localhost:8501` が開き、テキスト/画像で検索できます。

## 技術スタック

- **多言語 CLIP** (`open-clip-torch`, `xlm-roberta-base-ViT-B-32` / `laion5b_s13b_b90k`) — 画像・テキストの埋め込み。XLM-RoBERTa テキストタワーにより日本語クエリにも対応
- **FAISS** (`faiss-cpu`) — 高速な近傍検索
- **Streamlit** — Web UI

## ライセンス

MIT
