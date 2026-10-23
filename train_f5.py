# ============================================================

# PHAN CUM CO PHIEU - 3 BIEN RESILIENCE

# 1. Chon K: Ward, Elbow, Silhouette, CH

# 2. K-Means va Fuzzy C-Means

# 3. Danh gia WSS, BSS, Silhouette, CH, DBI

# 4. Tuning sieu tham so

# ============================================================


from pathlib import Path

from itertools import product

import warnings


import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import skfuzzy as fuzz


from scipy.cluster.hierarchy import linkage, dendrogram, fcluster


from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans

from sklearn.metrics import (

    silhouette_score,

    calinski_harabasz_score,

    davies_bouldin_score

)


warnings.filterwarnings("ignore")

# ============================================================
# 0. CAU HINH DUONG DAN
# ============================================================

from pathlib import Path

PROJECT_DIR = Path(
    r"G:\.shortcut-targets-by-id\1X9Wo6KizsPTddHOHc8jojsn0xmnVENC1"
        r"\[Nhóm 10] ML & AI trong tài chính"
    r"\ML_ChongChiu&PhucHoiCoPhieu"
)

# Thu muc luu dataset va bang ket qua
OUTPUT_DIR = (
    PROJECT_DIR / "data" / "processed" / "clustering_f5"
)

# Thu muc luu tat ca bieu do
FIGURE_DIR = (
    PROJECT_DIR / "reports" / "figures" / "clustering_f5"
)

# Tu dong tao hai thu muc neu chua ton tai
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# Tim file dau vao o cac vi tri co the co
INPUT_CANDIDATES = [
    PROJECT_DIR / "data" / "processed"
    / "preparation_preprocessing" / "df_scaled.csv",

    PROJECT_DIR / "data" / "processed" / "preparation_preprocessing" / "df_scaled.csv",
]

INPUT_FILE = next(
    (path for path in INPUT_CANDIDATES if path.exists()),
    None
)

if INPUT_FILE is None:
    raise FileNotFoundError(
        "Khong tim thay df_scaled.csv. Da kiem tra:\n"
        + "\n".join(str(p) for p in INPUT_CANDIDATES)
    )

print("File dau vao:", INPUT_FILE)
print("Thu muc luu dataset:", OUTPUT_DIR)
print("Thu muc luu bieu do:", FIGURE_DIR)


FEATURES = [
    "volatility_60d",
    "mean_return_60d",
    "momentum_60d",
    "avg_volume_60d",
    "beta",
]

RANDOM_STATE = 42

# Duong dan file du lieu dau vao
PROJECT_DIR = Path(
    r"G:\.shortcut-targets-by-id\1X9Wo6KizsPTddHOHc8jojsn0xmnVENC1"
    r"\[Nhóm 10] ML & AI trong tài chính"
    r"\ML_ChongChiu&PhucHoiCoPhieu"
)

INPUT_FILE = PROJECT_DIR / "data" / "processed" / "preparation_preprocessing" / "df_scaled.csv"

# Gioi han tim K

MAX_K_LIMIT = 10

TUNING_K_VALUES = [2, 3, 4, 5, 6]


# Tuning K-Means

INIT_VALUES = ["k-means++", "random"]

N_INIT_VALUES = [10, 50, 100]

MAX_ITER_VALUES = [100, 300, 500]

TOL_VALUES = [1e-4, 1e-3, 1e-2]


# Tuning Fuzzy C-Means

FCM_M_VALUES = [1.5, 2.0, 2.5, 3.0]

FCM_ERROR = 0.005

FCM_MAX_ITER = 1000


SHOW_PLOTS = True


pd.set_option("display.max_columns", None)

pd.set_option("display.width", 220)

pd.set_option("display.max_colwidth", 40)


def save_plot(filename):

    """Luu va hien thi bieu do."""

    path = FIGURE_DIR / filename

    plt.tight_layout()

    plt.savefig(path, dpi=300, bbox_inches="tight")

    if SHOW_PLOTS:

        plt.show()

    plt.close()

    print("Da luu bieu do:", path)


# ============================================================

# 1. DOC, LAM SACH VA CHUAN HOA DU LIEU

# ============================================================


if not INPUT_FILE.exists():

    raise FileNotFoundError(

        f"Khong tim thay file: {INPUT_FILE}"

    )


df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")


missing_columns = [

    col for col in FEATURES if col not in df.columns

]


if missing_columns:

    raise ValueError(

        f"File du lieu thieu cac cot: {missing_columns}"

    )


X_df = df[FEATURES].apply(

    pd.to_numeric, errors="coerce"

)


X_df = X_df.replace(

    [np.inf, -np.inf], np.nan

)


valid_mask = X_df.notna().all(axis=1)


df_valid = (

    df.loc[valid_mask]

    .copy()

    .reset_index(drop=True)

)


X_raw = X_df.loc[valid_mask].to_numpy(dtype=float)


if len(X_raw) < 4:

    raise ValueError(

        "Can it nhat 4 quan sat hop le de phan cum."

    )


if len(np.unique(X_raw, axis=0)) < 3:

    raise ValueError(

        "Khong du diem du lieu khac nhau de phan cum."

    )


if any(

    df_valid[col].nunique() < 2 for col in FEATURES

):

    raise ValueError(

        "Co bien khong co bien thien; can kiem tra du lieu."

    )


scaler = StandardScaler()

X = scaler.fit_transform(X_raw)


print("=" * 78)

print(

    "PHÂN CỤM VỚI 3 BIẾN: "

    "RELATIVE_RESILIENCE, RECOVERY_RATIO_60D, RESILIENCE"

)

print("=" * 78)


print("Số quan sát ban đầu:", len(df))

print("Số quan sát hợp lệ được dùng:", len(df_valid))

print(

    "Số quan sát bị loại do thiếu/không hợp lệ:",

    len(df) - len(df_valid)

)

print("Các biến đầu vào:", FEATURES)


print("\nThống kê mô tả 3 biến đầu vào:")

print(

    df_valid[FEATURES]

    .describe()

    .round(4)

    .to_string()

)


MAX_K = min(

    MAX_K_LIMIT,

    len(X) - 1

)


K_VALUES = list(range(2, MAX_K + 1))


TUNING_K_VALUES = sorted({

    k for k in TUNING_K_VALUES

    if 2 <= k <= MAX_K

})


if not TUNING_K_VALUES:

    TUNING_K_VALUES = K_VALUES.copy()


# ============================================================

# 2. HAM DANH GIA CHUNG

# ============================================================


def evaluate_clustering(X_data, labels):

    """

    WSS, BSS va cac chi so danh gia.

    WSS/BSS tinh theo nhan cum cuoi cung.

    """


    labels = np.asarray(labels)

    unique_labels = np.unique(labels)

    k = len(unique_labels)

    n = len(labels)


    if k < 2 or k >= n:

        return {

            "WSS": np.nan,

            "BSS": np.nan,

            "Silhouette": np.nan,

            "Calinski_Harabasz": np.nan,

            "Davies_Bouldin": np.nan

        }


    overall_mean = X_data.mean(axis=0)

    wss = 0.0

    bss = 0.0


    for cluster_id in unique_labels:

        cluster_data = X_data[labels == cluster_id]

        cluster_mean = cluster_data.mean(axis=0)


        wss += np.sum(

            (cluster_data - cluster_mean) ** 2

        )


        bss += len(cluster_data) * np.sum(

            (cluster_mean - overall_mean) ** 2

        )


    return {

        "WSS": wss,

        "BSS": bss,

        "Silhouette": silhouette_score(

            X_data, labels

        ),

        "Calinski_Harabasz": calinski_harabasz_score(

            X_data, labels

        ),

        "Davies_Bouldin": davies_bouldin_score(

            X_data, labels

        )

    }


def rank_results(table):

    """

    Silhouette: cao tot.

    CH: cao tot.

    DBI: thap tot.

    """


    table = table.copy()


    table["Rank_Silhouette"] = table[

        "Silhouette"

    ].rank(ascending=False, method="min")


    table["Rank_CH"] = table[

        "Calinski_Harabasz"

    ].rank(ascending=False, method="min")


    table["Rank_DBI"] = table[

        "Davies_Bouldin"

    ].rank(ascending=True, method="min")


    table["Total_Rank"] = table[

        ["Rank_Silhouette", "Rank_CH", "Rank_DBI"]

    ].sum(axis=1, min_count=3)


    return table.sort_values(

        ["Total_Rank", "Silhouette"],

        ascending=[True, False],

        na_position="last"

    ).reset_index(drop=True)


# ============================================================

# 3. CHON K BANG PHAN CAP WARD

# ============================================================


print("\n" + "=" * 78)

print("BUOC 1 - PHAN CUM PHAN CAP WARD")

print("=" * 78)


Z = linkage(

    X,

    method="ward",

    metric="euclidean"

)


plt.figure(figsize=(14, 7))


dendrogram(

    Z,

    truncate_mode="lastp",

    p=min(30, len(X)),

    show_leaf_counts=True,

    leaf_rotation=90

)


plt.title("Dendrogram - Hierarchical Clustering (Ward)")

plt.xlabel("Nhóm quan sát / số quan sát trong nhánh")

plt.ylabel("Khoảng cách liên kết")


save_plot("F5_hierarchical_dendrogram.png")


# Bước nhảy giữa hai khoảng cách gộp liên tiếp

heights = Z[:, 2]

gaps = np.diff(heights)

n_obs = len(X)


candidate_k = np.array([

    n_obs - (i + 1)

    for i in range(len(gaps))

])


in_range = (

    (candidate_k >= 2)

    & (candidate_k <= MAX_K)

)


if in_range.any():

    eligible_gaps = np.where(

        in_range, gaps, -np.inf

    )


    best_i = int(np.argmax(eligible_gaps))

    K_WARD = int(candidate_k[best_i])


    gap_table = pd.DataFrame({

        "K": candidate_k[in_range],

        "Khoang_cach_nhay": gaps[in_range]

    }).sort_values("K")


else:

    K_WARD = 2


    gap_table = pd.DataFrame(

        columns=["K", "Khoang_cach_nhay"]

    )


print("\nBảng bước nhảy khoảng cách:")

print(

    gap_table.round(4).to_string(index=False)

)


print("K gợi ý từ bước nhảy Ward:", K_WARD)


gap_table.to_csv(

    OUTPUT_DIR / "Ward_merge_gaps.csv",

    index=False,

    encoding="utf-8-sig"

)


if not gap_table.empty:

    plt.figure(figsize=(9, 5))


    plt.bar(

        gap_table["K"].astype(str),

        gap_table["Khoang_cach_nhay"]

    )


    plt.title("Bước nhảy khoảng cách Ward theo K")

    plt.xlabel("Số cụm K")

    plt.ylabel("Bước nhảy khoảng cách")

    plt.grid(axis="y", alpha=0.3)


    save_plot("F5_Ward_merge_gaps.png")


# ============================================================

# 4. ELBOW, SILHOUETTE VA CALINSKI-HARABASZ

# ============================================================


print("\n" + "=" * 78)

print("BUOC 2 - TIM K BANG ELBOW, SILHOUETTE VA CH")

print("=" * 78)


selection_rows = []


for k in K_VALUES:

    model = KMeans(

        n_clusters=k,

        init="k-means++",

        n_init=20,

        random_state=RANDOM_STATE

    )


    labels = model.fit_predict(X)


    selection_rows.append({

        "K": k,

        "WSS_Inertia": model.inertia_,

        "Silhouette": silhouette_score(X, labels),

        "Calinski_Harabasz": calinski_harabasz_score(

            X, labels

        )

    })


selection_table = pd.DataFrame(selection_rows)


print("\nBảng đánh giá lựa chọn K:")

print(

    selection_table.round(4).to_string(index=False)

)


selection_table.to_csv(

    OUTPUT_DIR / "choose_k_table.csv",

    index=False,

    encoding="utf-8-sig"

)


def estimate_elbow(ks, values):

    """Uoc luong diem khuyu bang khoang cach den duong noi hai dau."""

    ks = np.asarray(ks, dtype=float)

    values = np.asarray(values, dtype=float)


    if len(ks) < 3:

        return int(ks[0])


    if np.isclose(values[0], values[-1]):

        return int(ks[0])


    x = (ks - ks[0]) / (ks[-1] - ks[0])

    y = (

        (values - values[-1])

        / (values[0] - values[-1])

    )


    distance = np.abs(x + y - 1) / np.sqrt(2)


    # Khong chon hai dau duong cong lam diem khuyu

    distance[0] = -np.inf

    distance[-1] = -np.inf


    return int(ks[np.argmax(distance)])


K_ELBOW = estimate_elbow(

    selection_table["K"],

    selection_table["WSS_Inertia"]

)


K_SILHOUETTE = int(

    selection_table.loc[

        selection_table["Silhouette"].idxmax(), "K"

    ]

)


K_CH = int(

    selection_table.loc[

        selection_table["Calinski_Harabasz"].idxmax(), "K"

    ]

)


def plot_selection_curve(

    x, y, title, ylabel, filename, chosen_k

):

    plt.figure(figsize=(9, 5))

    plt.plot(x, y, marker="o")


    plt.axvline(

        chosen_k,

        linestyle="--",

        alpha=0.7,

        label=f"K đề xuất = {chosen_k}"

    )


    plt.title(title)

    plt.xlabel("Số cụm K")

    plt.ylabel(ylabel)

    plt.xticks(x)

    plt.grid(alpha=0.3)

    plt.legend()


    save_plot(filename)


plot_selection_curve(

    selection_table["K"],

    selection_table["WSS_Inertia"],

    "Elbow Method - K-Means",

    "WSS / Inertia",

    "F5_KMeans_Elbow.png",

    K_ELBOW

)


plot_selection_curve(

    selection_table["K"],

    selection_table["Silhouette"],

    "Silhouette Score - K-Means",

    "Silhouette Score",

    "F5_KMeans_Silhouette.png",

    K_SILHOUETTE

)


plot_selection_curve(

    selection_table["K"],

    selection_table["Calinski_Harabasz"],

    "Calinski-Harabasz Index - K-Means",

    "Calinski-Harabasz Index",

    "F5_KMeans_CH.png",

    K_CH

)


print("\nK từ các phương pháp:")

print("Ward:", K_WARD)

print("Elbow:", K_ELBOW)

print("Silhouette cao nhất:", K_SILHOUETTE)

print("Calinski-Harabasz cao nhất:", K_CH)


# Bốn phương pháp bỏ phiếu chọn K

k_votes = pd.Series([

    K_WARD,

    K_ELBOW,

    K_SILHOUETTE,

    K_CH

]).value_counts()


K_FINAL = int(

    k_votes[k_votes == k_votes.max()].index.min()

)


print("\nSố phiếu theo K:")

print(k_votes.sort_index().to_string())


print(

    "\nK_FINAL được chọn theo số phiếu cao nhất "

    "(nếu hòa, chọn K nhỏ hơn):",

    K_FINAL

)


if K_FINAL not in K_VALUES:

    raise ValueError("K_FINAL không nằm trong danh sách K hợp lệ.")


# ============================================================

# 5. TUNING K-MEANS

# ============================================================


print("\n" + "=" * 78)

print("BUOC 3 - TUNING SIEU THAM SO K-MEANS")

print("=" * 78)


km_rows = []


for k, init, n_init, max_iter, tol in product(

    TUNING_K_VALUES,

    INIT_VALUES,

    N_INIT_VALUES,

    MAX_ITER_VALUES,

    TOL_VALUES

):

    model = KMeans(

        n_clusters=k,

        init=init,

        n_init=n_init,

        max_iter=max_iter,

        tol=tol,

        random_state=RANDOM_STATE

    )


    labels = model.fit_predict(X)

    metrics = evaluate_clustering(X, labels)


    km_rows.append({

        "k": k,

        "init": init,

        "n_init": n_init,

        "max_iter": max_iter,

        "tol": tol,

        "n_iter_actual": model.n_iter_,

        "WSS_Inertia": model.inertia_,

        **metrics

    })


km_tuning_all = rank_results(

    pd.DataFrame(km_rows)

)


km_tuning_all.to_csv(

    OUTPUT_DIR / "KMeans_Hyperparameter_Tuning_All.csv",

    index=False,

    encoding="utf-8-sig"

)


# Xem top 10 cấu hình trên toàn bộ K

print("\nTOP 10 K-MEANS - TẤT CẢ GIÁ TRỊ K:")

print(

    km_tuning_all.head(10)

    .round(5)

    .to_string(index=False)

)


# Chọn cấu hình tốt nhất với K_FINAL để so sánh công bằng

km_fixed_k = km_tuning_all[

    km_tuning_all["k"] == K_FINAL

].copy()


if km_fixed_k.empty:

    raise ValueError(

        f"Không có cấu hình K-Means với K_FINAL={K_FINAL}. "

        "Hãy bổ sung K_FINAL vào TUNING_K_VALUES."

    )


km_fixed_k = rank_results(km_fixed_k)

best_km = km_fixed_k.iloc[0]


print("\nCẤU HÌNH K-MEANS TỐT NHẤT VỚI K_FINAL:")

print(best_km.to_frame("Gia_tri").to_string())


km_tuning_all[km_tuning_all["k"] == K_FINAL].to_csv(

    OUTPUT_DIR / "KMeans_Tuning_Selected_K.csv",

    index=False,

    encoding="utf-8-sig"

)


# ============================================================

# 6. TUNING FUZZY C-MEANS

# ============================================================


print("\n" + "=" * 78)

print("BUOC 4 - TUNING SIEU THAM SO FUZZY C-MEANS")

print("=" * 78)


fcm_rows = []


for k, m in product(

    TUNING_K_VALUES,

    FCM_M_VALUES

):

    (

        centers,

        membership,

        initial_membership,

        distances,

        objective_history,

        iterations,

        fpc

    ) = fuzz.cluster.cmeans(

        data=X.T,

        c=k,

        m=m,

        error=FCM_ERROR,

        maxiter=FCM_MAX_ITER,

        seed=RANDOM_STATE

    )


    labels = np.argmax(membership, axis=0)


    metrics = evaluate_clustering(X, labels)


    fcm_rows.append({

        "k": k,

        "m": m,

        "iterations_actual": iterations,

        "objective_final": objective_history[-1],

        "FPC": fpc,

        **metrics

    })


fcm_tuning_all = rank_results(

    pd.DataFrame(fcm_rows)

)


fcm_tuning_all.to_csv(

    OUTPUT_DIR / "FCM_Hyperparameter_Tuning_All.csv",

    index=False,

    encoding="utf-8-sig"

)


print("\nTOP 10 FUZZY C-MEANS - TẤT CẢ GIÁ TRỊ K:")

print(

    fcm_tuning_all.head(10)

    .round(5)

    .to_string(index=False)

)


fcm_fixed_k = fcm_tuning_all[

    fcm_tuning_all["k"] == K_FINAL

].copy()


if fcm_fixed_k.empty:

    raise ValueError(

        f"Không có cấu hình FCM với K_FINAL={K_FINAL}. "

        "Hãy bổ sung K_FINAL vào TUNING_K_VALUES."

    )


fcm_fixed_k = rank_results(fcm_fixed_k)

# Chọn đồng thời bộ siêu tham số tốt nhất (K_FINAL, m).
# Vì K_FINAL đã được chọn ở bước trước, m được chọn theo thứ hạng
# tổng hợp Silhouette, Calinski-Harabasz và Davies-Bouldin.
best_fcm = fcm_fixed_k.iloc[0]
BEST_FCM_M = float(best_fcm["m"])

print("\nBẢNG TUNING FUZZY C-MEANS TẠI K_FINAL:")
print(
    fcm_fixed_k.round(5).to_string(index=False)
)

print("\nCẤU HÌNH FUZZY C-MEANS TỐT NHẤT VỚI K_FINAL:")
print(best_fcm.to_frame("Gia_tri").to_string())
print(f"\nHệ số m được chọn cho mô hình Fuzzy C-Means: {BEST_FCM_M}")

pd.DataFrame([best_fcm]).to_csv(
    OUTPUT_DIR / "FCM_Best_Hyperparameters.csv",
    index=False,
    encoding="utf-8-sig"
)


fcm_tuning_all[fcm_tuning_all["k"] == K_FINAL].to_csv(

    OUTPUT_DIR / "FCM_Tuning_Selected_K.csv",

    index=False,

    encoding="utf-8-sig"

)


# ============================================================

# 7. CHAY HAI MO HINH CUOI CUNG VOI K_FINAL

# ============================================================


print("\n" + "=" * 78)

print(f"BUOC 5 - CHAY HAI MO HINH CUOI CUNG: K = {K_FINAL}")

print("=" * 78)


# K-Means

final_kmeans = KMeans(

    n_clusters=K_FINAL,

    init=best_km["init"],

    n_init=int(best_km["n_init"]),

    max_iter=int(best_km["max_iter"]),

    tol=float(best_km["tol"]),

    random_state=RANDOM_STATE

)


df_valid["kmeans_cluster"] = (

    final_kmeans.fit_predict(X) + 1

)


print("\nK-Means - số quan sát mỗi cụm:")

print(

    df_valid["kmeans_cluster"]

    .value_counts()

    .sort_index()

    .to_string()

)


# Fuzzy C-Means

(

    fcm_centers,

    membership,

    initial_membership,

    distances,

    objective_history,

    fcm_iterations,

    fpc

) = fuzz.cluster.cmeans(

    data=X.T,

    c=K_FINAL,

    m=BEST_FCM_M,

    error=FCM_ERROR,

    maxiter=FCM_MAX_ITER,

    seed=RANDOM_STATE

)


df_valid["fuzzy_cluster"] = (

    np.argmax(membership, axis=0) + 1

)


df_valid["fuzzy_max_membership"] = (

    membership.max(axis=0)

)


for i in range(K_FINAL):

    df_valid[f"fuzzy_membership_cluster_{i+1}"] = (

        membership[i]

    )


    df_valid[f"fuzzy_cluster_{i+1}_pct"] = (

        membership[i] * 100

    ).round(2)


print("\nFuzzy C-Means - số quan sát mỗi cụm:")

print(

    df_valid["fuzzy_cluster"]

    .value_counts()

    .sort_index()

    .to_string()

)


print("\nFuzzy Partition Coefficient (FPC):", round(fpc, 5))

print("Số vòng lặp FCM:", fcm_iterations)
print("Hệ số m được sử dụng:", BEST_FCM_M)


# ============================================================
# 8. SO SANH K-MEANS VA FUZZY C-MEANS
# ============================================================

print("\n" + "=" * 100)
print("BUOC 6 - SO SANH HAI MO HINH PHAN CUM")
print("=" * 100)

# So sánh chỉ số đánh giá

km_labels = df_valid["kmeans_cluster"].to_numpy()
fcm_labels = df_valid["fuzzy_cluster"].to_numpy()

km_metrics = evaluate_clustering(X, km_labels)
fcm_metrics = evaluate_clustering(X, fcm_labels)

comparison_metrics = pd.DataFrame([
    {"Mo_hinh": "K-Means", **km_metrics},
    {"Mo_hinh": "Fuzzy C-Means", **fcm_metrics}
])

print("\n--- 8.1. SO SANH CHI SO DANH GIA ---")
print("Luu y: Fuzzy C-Means duoc danh gia theo cum cung "
      "(nhan co membership cao nhat).")

print(comparison_metrics.round(5).to_string(index=False))

comparison_metrics.to_csv(
    OUTPUT_DIR / "So_sanh_chi_so_KMeans_FCM.csv",
    index=False,
    encoding="utf-8-sig"
)

# In lai cau hinh tot nhat cua tung mo hinh
print("\n--- CAU HINH K-MEANS DA CHON ---")
print(best_km.to_frame("Gia_tri").to_string())

print("\n--- CAU HINH FUZZY C-MEANS DA CHON ---")
print(best_fcm.to_frame("Gia_tri").to_string())


# ============================================================
# 9. IN KET QUA K-MEANS THEO TUNG GIAI DOAN
# ============================================================

print("\n" + "=" * 100)
print("BUOC 7 - KET QUA PHAN CUM K-MEANS THEO GIAI DOAN")
print("=" * 100)

if "ticker" in df_valid.columns and "stage_name" in df_valid.columns:

    stage_cols = ["stage_name"]

    if "start_date" in df_valid.columns:
        stage_cols.append("start_date")

    for stage_key, stage_df in df_valid.groupby(
        stage_cols, dropna=False, sort=False
    ):
        if not isinstance(stage_key, tuple):
            stage_key = (stage_key,)

        print("\n" + "=" * 100)
        print(f"GIAI DOAN: {stage_key[0]}")

        if len(stage_key) > 1:
            if pd.notna(stage_key[1]):
                date_text = pd.to_datetime(
                    stage_key[1]
                ).strftime("%Y-%m-%d")
            else:
                date_text = "Khong co thong tin"

            print(f"Ngay bat dau: {date_text}")

        print("=" * 100)
        print("\n--- K-MEANS ---")

        for cluster_id in range(1, K_FINAL + 1):
            cluster_df = stage_df[
                stage_df["kmeans_cluster"] == cluster_id
            ]

            tickers = sorted(
                cluster_df["ticker"].dropna().astype(str).unique()
            )

            if tickers:
                print(
                    f"\nCum {cluster_id} ({len(tickers)} ma): "
                    + ", ".join(tickers)
                )
            else:
                print(f"\nCum {cluster_id} (0 ma): Khong co ma")

            if not cluster_df.empty:
                print("Dac trung trung binh cua cum:")

                print(
                    cluster_df[FEATURES]
                    .mean()
                    .to_frame("Gia_tri_trung_binh")
                    .round(4)
                    .to_string()
                )

    # Luu ket qua tung ma cua K-Means
    km_result_cols = (
        stage_cols
        + ["ticker", "kmeans_cluster"]
        + FEATURES
    )

    df_valid[km_result_cols].to_csv(
        OUTPUT_DIR / "KMeans_Ket_qua_tung_ma_theo_giai_doan.csv",
        index=False,
        encoding="utf-8-sig"
    )

else:
    print(
        "Khong the in ket qua theo giai doan: "
        "thieu cot ticker hoac stage_name."
    )


# ============================================================
# 10. IN KET QUA FUZZY C-MEANS THEO TUNG GIAI DOAN
# ============================================================

print("\n" + "=" * 100)
print("BUOC 8 - KET QUA PHAN CUM FUZZY C-MEANS THEO GIAI DOAN")
print(f"So cum K = {K_FINAL} | He so m toi uu = {BEST_FCM_M}")
print("=" * 100)

if "ticker" in df_valid.columns and "stage_name" in df_valid.columns:

    # Tao bang ket qua chi tiet
    fcm_result = df_valid.copy()

    fcm_result["fuzzy_max_membership_pct"] = (
        fcm_result["fuzzy_max_membership"] * 100
    ).round(2)

    membership_pct_cols = [
        f"fuzzy_cluster_{i}_pct"
        for i in range(1, K_FINAL + 1)
    ]

    result_cols = (
        stage_cols
        + [
            "ticker",
            "fuzzy_cluster",
            "fuzzy_max_membership_pct"
        ]
        + FEATURES
        + membership_pct_cols
    )

    for stage_key, stage_df in fcm_result.groupby(
        stage_cols, dropna=False, sort=False
    ):
        if not isinstance(stage_key, tuple):
            stage_key = (stage_key,)

        print("\n" + "=" * 100)
        print(f"GIAI DOAN: {stage_key[0]}")

        if len(stage_key) > 1:
            if pd.notna(stage_key[1]):
                date_text = pd.to_datetime(
                    stage_key[1]
                ).strftime("%Y-%m-%d")
            else:
                date_text = "Khong co thong tin"

            print(f"Ngay bat dau: {date_text}")

        print("=" * 100)

        # ----------------------------------------------------
        # 10.1. CHI TIET TUNG MA VA TY LE THUOC CAC CUM MEM
        # ----------------------------------------------------
        print("\n--- 1. KET QUA CHI TIET TUNG MA CO PHIEU ---")
        print(
            "fuzzy_cluster la cum cung; "
            "fuzzy_max_membership_pct la ty le thuoc cum cung."
        )

        print(
            stage_df[result_cols]
            .sort_values(["fuzzy_cluster", "ticker"])
            .round(4)
            .to_string(index=False)
        )

        # ----------------------------------------------------
        # 10.2. DAC TRUNG CUM CUNG
        # ----------------------------------------------------
        print("\n--- 2. DAC TRUNG CAC CUM CUNG ---")

        for cluster_id in range(1, K_FINAL + 1):
            cluster_df = stage_df[
                stage_df["fuzzy_cluster"] == cluster_id
            ]

            tickers = sorted(
                cluster_df["ticker"].dropna().astype(str).unique()
            )

            if tickers:
                print(
                    f"\nCum cung {cluster_id} ({len(tickers)} ma): "
                    + ", ".join(tickers)
                )
                print("Dac trung trung binh cua cum:")

                print(
                    cluster_df[FEATURES]
                    .mean()
                    .to_frame("Gia_tri_trung_binh")
                    .round(4)
                    .to_string()
                )
            else:
                print(f"\nCum cung {cluster_id} (0 ma): Khong co ma")

        # ----------------------------------------------------
        # 10.3. DAC TRUNG CUM MEM
        # ----------------------------------------------------
        print("\n--- 3. DAC TRUNG CAC CUM MEM ---")
        print(
            "Trung binh co trong so membership cua tung ma; "
            f"su dung m = {BEST_FCM_M}."
        )

        for cluster_id in range(1, K_FINAL + 1):
            # Dung membership chua lam tron lam trong so
            membership_col = f"fuzzy_membership_cluster_{cluster_id}"
            weights = stage_df[membership_col].to_numpy()

            print(f"\nCum mem {cluster_id}:")

            soft_profile = {}

            for feature in FEATURES:
                values = stage_df[feature].to_numpy(dtype=float)
                valid = np.isfinite(values) & np.isfinite(weights)

                if valid.any() and weights[valid].sum() > 0:
                    soft_profile[feature] = np.average(
                        values[valid],
                        weights=weights[valid]
                    )
                else:
                    soft_profile[feature] = np.nan

            print(
                pd.Series(
                    soft_profile,
                    name="Trung_binh_co_trong_so"
                ).round(4).to_string()
            )

        # ----------------------------------------------------
        # 10.4. LUU DATASET DAY DU SAU PHAM CUM
        # ----------------------------------------------------

        df_valid.to_csv(
            OUTPUT_DIR / "Dataset_day_du_KMeans_FCM_F5.csv",
            index=False,
        encoding="utf-8-sig"
        )

        print(
            "Đã lưu dataset đầy đủ:",
            OUTPUT_DIR / "Dataset_day_du_KMeans_FCM_F5.csv"
        )

# ============================================================
# 11. LUU BANG SO SANH TONG HOP CUOI CUNG
# ============================================================

final_comparison = comparison_metrics.copy()

final_comparison.to_csv(
    OUTPUT_DIR / "Tong_hop_so_sanh_KMeans_FCM_F5.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nBẢNG SO SÁNH K-MEANS VÀ FUZZY C-MEANS:")
print(final_comparison.round(5).to_string(index=False))

print("\nĐã lưu bảng so sánh tại:")
print(OUTPUT_DIR / "Tong_hop_so_sanh_KMeans_FCM_F5.csv")

# ============================================================
# 12. TRUC QUAN HOA KẾT QUẢ PHAN CUM
# ============================================================

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

# 12.1. Chọn các giai đoạn muốn trực quan hóa
STAGES_TO_PLOT = [
    "Bùng phát đại dịch COVID-19 toàn cầu (Ca bệnh số 17 & Phong tỏa)",
    "Khủng hoảng Trái phiếu doanh nghiệp, Sự cố SCB - Vạn Thịnh Phát"
]

# 12.2. Chuẩn bị dữ liệu
df_plot = df_valid.copy().reset_index(drop=True)

X_array = np.asarray(X)

if len(X_array) != len(df_plot):
    raise ValueError("Số dòng của X và df_valid không khớp.")

# PCA từ đúng các biến đã dùng để phân cụm
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_array)

df_plot["PCA1"] = X_pca[:, 0]
df_plot["PCA2"] = X_pca[:, 1]

# Các cột membership của Fuzzy C-Means
membership_cols = [
    f"fuzzy_cluster_{i}_pct"
    for i in range(1, K_FINAL + 1)
]

missing_cols = [
    col for col in membership_cols
    if col not in df_plot.columns
]

if missing_cols:
    raise ValueError(f"Thiếu cột membership: {missing_cols}")

sns.set_theme(style="whitegrid")

# 12.3. Lặp qua từng giai đoạn
for STAGE_TO_PLOT in STAGES_TO_PLOT:

    df_stage = df_plot[
        df_plot["stage_name"].astype(str) == STAGE_TO_PLOT
    ].copy()

    if df_stage.empty:
        print(f"Không tìm thấy giai đoạn: {STAGE_TO_PLOT}")
        print("Các giai đoạn hiện có:")
        print(df_plot["stage_name"].dropna().unique())
        continue

    # 12.4. Tạo 3 biểu đồ
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    # --------------------------------------------------------
    # Biểu đồ 1: K-Means - Cụm cứng
    # --------------------------------------------------------
    sns.scatterplot(
        data=df_stage,
        x="PCA1",
        y="PCA2",
        hue="kmeans_cluster",
        palette="Set1",
        s=130,
        ax=axes[0]
    )

    for _, row in df_stage.iterrows():
        axes[0].annotate(
            str(row["ticker"]),
            (row["PCA1"], row["PCA2"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8
        )

    axes[0].set_title("K-Means - Cụm cứng")
    axes[0].set_xlabel("PCA 1")
    axes[0].set_ylabel("PCA 2")
    axes[0].legend(title="Cụm")

    # --------------------------------------------------------
    # Biểu đồ 2: Fuzzy C-Means - Cụm cứng
    # --------------------------------------------------------
    sns.scatterplot(
        data=df_stage,
        x="PCA1",
        y="PCA2",
        hue="fuzzy_cluster",
        palette="Set1",
        s=130,
        ax=axes[1]
    )

    for _, row in df_stage.iterrows():
        axes[1].annotate(
            str(row["ticker"]),
            (row["PCA1"], row["PCA2"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8
        )

    axes[1].set_title("Fuzzy C-Means - Cụm cứng")
    axes[1].set_xlabel("PCA 1")
    axes[1].set_ylabel("PCA 2")
    axes[1].legend(title="Cụm")

    # --------------------------------------------------------
    # Biểu đồ 3: Fuzzy C-Means - Cụm mềm
    # --------------------------------------------------------
    df_membership = df_stage.set_index("ticker")[
        membership_cols
    ].copy()

    df_membership.columns = [
        f"Cụm mềm {i}"
        for i in range(1, K_FINAL + 1)
    ]

    df_membership.plot(
        kind="bar",
        stacked=True,
        ax=axes[2],
        colormap="Set2",
        width=0.8
    )

    axes[2].set_title(
        "Fuzzy C-Means - Mức độ thuộc cụm mềm"
    )
    axes[2].set_xlabel("Mã cổ phiếu")
    axes[2].set_ylabel("Mức độ thuộc (%)")
    axes[2].set_ylim(0, 100)
    axes[2].tick_params(
        axis="x",
        labelrotation=45
    )
    axes[2].legend(
        title="Cụm",
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )

    # --------------------------------------------------------
    # 12.5. Tiêu đề và lưu biểu đồ riêng từng giai đoạn
    # --------------------------------------------------------
    fig.suptitle(
        f"So sánh K-Means và Fuzzy C-Means\n"
        f"Giai đoạn: {STAGE_TO_PLOT}",
        fontsize=14
    )

    fig.tight_layout(rect=[0, 0, 1, 0.90])

    # Đặt tên file theo từng giai đoạn
    if "COVID-19" in STAGE_TO_PLOT:
        filename = "F5_COVID19_KMeans_FCM.png"
        csv_filename = "Membership_COVID19.csv"
    else:
        filename = "F5_SC B_VanThinhPhat_KMeans_FCM.png".replace(
            " ", ""
        )
        csv_filename = "Membership_SC B_VanThinhPhat.csv".replace(
            " ", ""
        )

    save_plot(filename)

    # Lưu bảng membership của từng giai đoạn
    membership_export = df_stage[
        [
            "stage_name",
            "ticker",
            "kmeans_cluster",
            "fuzzy_cluster"
        ] + membership_cols
    ].copy()

    membership_export.to_csv(
        OUTPUT_DIR / csv_filename,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\nĐã hoàn thành giai đoạn: {STAGE_TO_PLOT}")
    print(f"Đã lưu biểu đồ: {filename}")
    print(
        "Đã lưu bảng membership:",
        OUTPUT_DIR / csv_filename
    )

    plt.show()
    plt.close(fig)

# 12.6. In tỷ lệ phương sai PCA
print(
    "\nTổng phương sai được giải thích bởi 2 thành phần PCA:",
    round(pca.explained_variance_ratio_.sum() * 100, 2),
    "%"
)


# ============================================================
# LUU MO HINH F5 CHO STREAMLIT
# ============================================================

import joblib

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Lưu K-Means F5
joblib.dump(
    {
        "model": final_kmeans,
        "features": FEATURES,
        "n_clusters": K_FINAL,
    },
    MODEL_DIR / "kmeans_f5.pkl"
)

# Lưu Fuzzy C-Means F5
joblib.dump(
    {
        "centers": fcm_centers,
        "m": BEST_FCM_M,
        "features": FEATURES,
        "n_clusters": K_FINAL,
    },
    MODEL_DIR / "fcm_f5.pkl"
)

print("Đã lưu kmeans_f5.pkl và fcm_f5.pkl")