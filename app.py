import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # faiss/torch の OpenMP 競合回避

import io
import streamlit as st
from PIL import Image

from encoder import CLIPEncoder
from indexer import FAISSIndexer

INDEX_DIR = "index"
TOP_K = 12

st.set_page_config(page_title="ビジュアル類似性検索", page_icon="🔍", layout="wide")


@st.cache_resource  # モデル・インデックスは初回のみロード
def load_resources():
    encoder = CLIPEncoder()
    indexer = FAISSIndexer()
    indexer.load(INDEX_DIR)
    return encoder, indexer


encoder, indexer = load_resources()

st.title("🔍 ビジュアル類似性検索")
st.caption(f"CLIP × FAISS — {indexer.total:,} 枚のインデックス")

# 検索モード
mode = st.radio("検索モード", ["画像から検索", "テキストから検索"], horizontal=True)

query_vector = None

if mode == "画像から検索":
    uploaded = st.file_uploader("クエリ画像をアップロード", type=["jpg", "jpeg", "png", "webp"])
    if uploaded:
        query_image = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
        col_preview, _ = st.columns([1, 3])
        with col_preview:
            st.image(query_image, caption="クエリ画像", use_container_width=True)
        query_vector = encoder.encode_images(query_image)

else:
    text_input = st.text_input(
        "検索テキスト（日本語・英語どちらも可）",
        placeholder="例: red sneakers, 白い犬, mountain landscape",
    )
    if text_input:
        query_vector = encoder.encode_text(text_input)

# 検索実行
if query_vector is not None:
    with st.spinner("検索中..."):
        results = indexer.search(query_vector, top_k=TOP_K)

    st.markdown("---")
    st.subheader(f"検索結果（上位 {len(results)} 件）")

    cols = st.columns(4)
    for i, result in enumerate(results):
        with cols[i % 4]:
            try:
                img = Image.open(result["abs_path"])
                st.image(img, use_container_width=True)
                score_pct = result["score"] * 100
                # スコアに応じてバーを色分け
                bar_color = "#1D9E75" if score_pct > 70 else "#EF9F27" if score_pct > 50 else "#888780"
                st.markdown(
                    f'<div style="height:3px;background:{bar_color};border-radius:2px;margin:2px 0 4px"></div>'
                    f'<small style="color:#888">{score_pct:.1f}%　{result["filename"]}</small>',
                    unsafe_allow_html=True,
                )
            except FileNotFoundError:
                st.warning(f"ファイルが見つかりません: {result['filename']}")