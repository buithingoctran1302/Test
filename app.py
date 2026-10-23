
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
DATA_FILE = (
    ROOT / "data" / "processed"
    / "preparation_preprocessing" / "df_scaled.csv"
)

st.set_page_config(
    page_title="Phân cụm cổ phiếu",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Phân cụm khả năng chống chịu và phục hồi cổ phiếu")
st.caption("Ứng dụng so sánh K-Means và Fuzzy C-Means")

if not DATA_FILE.exists():
    st.error(f"Không tìm thấy dữ liệu: {DATA_FILE}")
    st.stop()

df = pd.read_csv(DATA_FILE)

col1, col2 = st.columns(2)

with col1:
    feature_set = st.selectbox(
        "Bộ biến đầu vào",
        ["F3 - 3 biến chống chịu/phục hồi",
         "F5 - 5 biến đặc trưng"]
    )

with col2:
    algorithm = st.selectbox(
        "Thuật toán",
        ["K-Means", "Fuzzy C-Means"]
    )

model_key = "f3" if feature_set.startswith("F3") else "f5"
model_name = (
    f"kmeans_{model_key}.pkl"
    if algorithm == "K-Means"
    else f"fcm_{model_key}.pkl"
)
model_path = MODEL_DIR / model_name

if not model_path.exists():
    st.error(f"Không tìm thấy mô hình: {model_path}")
    st.stop()

artifact = joblib.load(model_path)
features = artifact["features"]

missing = [c for c in features if c not in df.columns]
if missing:
    st.error(f"Dữ liệu thiếu các biến: {missing}")
    st.stop()

X = df[features].apply(pd.to_numeric, errors="coerce")
valid_mask = X.notna().all(axis=1)
df_result = df.loc[valid_mask].copy()
X_valid = X.loc[valid_mask].to_numpy(dtype=float)

if df_result.empty:
    st.error("Không có dòng dữ liệu hợp lệ để phân cụm.")
    st.stop()

if algorithm == "K-Means":
    model = artifact["model"]
    df_result["cluster"] = model.predict(X_valid) + 1
else:
    centers = np.asarray(artifact["centers"], dtype=float)
    m = float(artifact["m"])

    if m <= 1:
        st.error("Tham số m của Fuzzy C-Means phải lớn hơn 1.")
        st.stop()

    distances = np.linalg.norm(
        X_valid[:, None, :] - centers[None, :, :],
        axis=2
    )
    distances = np.maximum(distances, 1e-10)
    inverse_distances = distances ** (-2.0 / (m - 1.0))
    membership = inverse_distances / inverse_distances.sum(
        axis=1, keepdims=True
    )

    df_result["cluster"] = np.argmax(membership, axis=1) + 1
    df_result["max_membership"] = membership.max(axis=1)

st.subheader("Tổng quan dữ liệu")

c1, c2, c3 = st.columns(3)
c1.metric("Số quan sát", len(df_result))
c2.metric("Số mã cổ phiếu", df_result["ticker"].nunique())
c3.metric("Số cụm", df_result["cluster"].nunique())

if algorithm == "Fuzzy C-Means":
    st.info(
        "Cột max_membership thể hiện mức độ thuộc cụm cao nhất "
        "của mỗi quan sát."
    )

st.subheader("Số quan sát trong từng cụm")
cluster_counts = (
    df_result["cluster"]
    .value_counts()
    .sort_index()
)
st.bar_chart(cluster_counts)

st.subheader("Kết quả phân cụm")

if "stage_name" in df_result.columns:
    stages = ["Tất cả"] + sorted(
        df_result["stage_name"].dropna().unique().tolist()
    )
    selected_stage = st.selectbox("Lọc theo giai đoạn", stages)

    if selected_stage != "Tất cả":
        df_result = df_result[
            df_result["stage_name"] == selected_stage
        ]

st.dataframe(df_result, use_container_width=True)

st.download_button(
    "Tải kết quả phân cụm (CSV)",
    data=df_result.to_csv(index=False).encode("utf-8-sig"),
    file_name=f"ket_qua_{model_key}_{algorithm.lower().replace(' ', '_')}.csv",
    mime="text/csv"
)