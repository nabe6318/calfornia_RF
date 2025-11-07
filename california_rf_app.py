# California Housing × RandomForest 回帰アプリ / Streamlit
# - 先頭行の表示（既定50）
# - 2変数を選んで予測ヒートマップ（他特徴量は中央値で固定）
# - ランダムフォレストのハイパーパラメータ調整、評価（R2 / RMSE / MAE / CV R2）、重要度
# -----------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.datasets import fetch_california_housing

st.set_page_config(page_title="California Housing × RandomForest", layout="wide")

# 0) データセットの説明（大学生向け）
st.markdown(
    """
    <h3 style="font-size:22px; margin-bottom:8px;">
    🏠 California Housing × RandomForest（回帰）雑草研・システム研　統計ゼミ
    </h3>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    本アプリは **California Housing**（米カリフォルニア州・18940地区）の住宅価格（中央値, ×100,000 USD）を  
    **ランダムフォレスト（RandomForest 回帰）** で予測・可視化します。  
    ランダムフォレストは多数の決定木を**乱択**して学習し、**アンサンブル平均**で精度と汎化性能を高める手法です。
    """
)

# 1) データ読み込み
cal = fetch_california_housing(as_frame=True)
X_full = cal.data.copy()
y = cal.target.copy()
feature_names = list(X_full.columns)

# 2) サイドバー設定
st.sidebar.header("⚙️ 学習設定 / Controls")
show_rows = st.sidebar.number_input("表示行数 / Rows to show", 10, len(X_full), 50, 10)

split_ratio = st.sidebar.slider("学習データの割合 / Train size", 0.5, 0.9, 0.8, 0.05)
random_state = st.sidebar.number_input("乱数シード / Random state", 0, 9999, 42, 1)

selected_features = st.sidebar.multiselect("特徴量の選択 / Select features", feature_names, default=feature_names)
if len(selected_features) < 2:
    st.sidebar.warning("少なくとも2つの特徴量を選択してください。")

axis_opts = selected_features if selected_features else feature_names
x_axis = st.sidebar.selectbox("X軸", axis_opts, index=0)
y_axis_opts = [c for c in axis_opts if c != x_axis] or [c for c in feature_names if c != x_axis]
y_axis = st.sidebar.selectbox("Y軸", y_axis_opts, index=0)

st.sidebar.subheader("🌲 ランダムフォレスト・パラメータ")
n_estimators = st.sidebar.slider("n_estimators（木の本数）", 10, 500, 200, 10)
max_depth = st.sidebar.slider("最大深さ / max_depth（0=制限なし）", 0, 30, 0, 1)
max_depth_arg = None if max_depth == 0 else max_depth
min_samples_split = st.sidebar.slider("min_samples_split", 2, 50, 10, 1)
min_samples_leaf = st.sidebar.slider("min_samples_leaf", 1, 50, 2, 1)
max_features = st.sidebar.selectbox("max_features", ["auto", "sqrt", "log2", "all"], index=1)
max_features_arg = None if max_features == "all" else max_features
bootstrap = st.sidebar.checkbox("bootstrap", value=True)
oob_score = st.sidebar.checkbox("OOB スコア（bootstrap時のみ）", value=False)

cv_k = st.sidebar.slider("交差検証分割数 / CV folds", 2, 10, 5, 1)

# 3) 先頭行の確認
st.markdown("### 1) データの確認（先頭行）")
st.dataframe(pd.concat([X_full, y.rename("MedHouseVal")], axis=1).head(show_rows), use_container_width=True)
st.caption("スケールや分布の雰囲気をつかみます。")

# 4) 学習と評価
X = X_full[selected_features].values if selected_features else X_full.values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=split_ratio, random_state=random_state
)

rf = RandomForestRegressor(
    n_estimators=n_estimators,
    max_depth=max_depth_arg,
    min_samples_split=min_samples_split,
    min_samples_leaf=min_samples_leaf,
    max_features=max_features_arg,      # None=全特徴量
    bootstrap=bootstrap,
    oob_score=oob_score and bootstrap,  # OOBはbootstrapがTrueのときのみ
    n_jobs=-1,
    random_state=random_state,
)
rf.fit(X_train, y_train)

# 交差検証（R2）
cv_r2 = cross_val_score(rf, X, y, cv=cv_k, scoring="r2", n_jobs=-1)

# テスト評価（rmseは後方互換）
pred = rf.predict(X_test)
try:
    rmse = mean_squared_error(y_test, pred, squared=False)
except TypeError:
    rmse = np.sqrt(mean_squared_error(y_test, pred))
mae = mean_absolute_error(y_test, pred)
r2 = r2_score(y_test, pred)

left, right = st.columns([1.1, 1])

with left:
    st.markdown("### 2) 評価 / Evaluation")
    oob_txt = f"  |  **OOB R²:** {getattr(rf, 'oob_score_', np.nan):.3f}" if (bootstrap and oob_score) else ""
    st.write(f"**R² (test):** {r2:.3f}  |  **RMSE:** {rmse:.3f}  |  **MAE:** {mae:.3f}{oob_txt}")
    st.write(f"**CV R² mean:** {cv_r2.mean():.3f}  (± {cv_r2.std():.3f})")

    # 2D ヒートマップ（他特徴量は中央値で固定）
    st.markdown("### 3) 2変数でみる予測ヒートマップ（他変数=中央値）")
    if x_axis and y_axis:
        base = X_full[selected_features].median() if selected_features else X_full.median()
        x_vals = np.linspace(X_full[x_axis].min(), X_full[x_axis].max(), 150)
        y_vals = np.linspace(X_full[y_axis].min(), X_full[y_axis].max(), 150)
        xx, yy = np.meshgrid(x_vals, y_vals)
        grid = pd.DataFrame({col: np.full(xx.size, base[col] if col in base.index else X_full[col].median())
                             for col in (selected_features if selected_features else feature_names)})
        grid[x_axis] = xx.ravel()
        grid[y_axis] = yy.ravel()

        Z = rf.predict(grid.values).reshape(xx.shape)
        fig_hm, ax_hm = plt.subplots(figsize=(7, 5.2), dpi=140)
        hm = ax_hm.contourf(xx, yy, Z, levels=18, alpha=0.9)
        cbar = fig_hm.colorbar(hm, ax=ax_hm, fraction=0.046, pad=0.04)
        cbar.set_label("Predicted MedHouseVal (×100k USD)")
        ax_hm.set_xlabel(x_axis)
        ax_hm.set_ylabel(y_axis)
        ax_hm.set_title("RandomForest prediction heatmap")
        st.pyplot(fig_hm, use_container_width=True)

with right:
    st.markdown("### 4) 特徴量の重要度 / Feature importances")
    importances = pd.Series(rf.feature_importances_, index=(selected_features if selected_features else feature_names))
    st.dataframe(importances.sort_values(ascending=False).to_frame("importance"))

with st.expander("🧠 RandomForest の詳細解説（大学授業向け）"):
    st.markdown(
        """
        ### 🌲 ランダムフォレストとは？
        **ランダムフォレスト（Random Forest）** は、  
        「**たくさんの決定木を作って、それらの結果を平均（回帰）または多数決（分類）する**」手法です。  
        つまり、**“森”のように多くの木を使って判断する**ことで、1本の木よりも安定した予測を行います。

        ---
        ### 🧩 どうして「ランダム」なのか？
        - 各決定木は、**訓練データをランダムに抽出（ブートストラップ法）**して学習します。  
        - さらに、各分岐（ノード）では **使う特徴量もランダムに選びます**。  
        👉 この“ランダム性”により、木ごとの個性が生まれ、**全体として偏らないモデル**になります。

        ---
        ### 💪 メリットと特徴
        | 特徴 | 内容 |
        |------|------|
        | **過学習しにくい** | 木を多数平均するため、ノイズの影響が小さい |
        | **汎化性能が高い** | 未知データにも比較的強い（安定した予測） |
        | **精度が高い** | 複雑な非線形関係も学習可能 |
        | **重要度がわかる** | 各特徴量の「どれくらい効いているか」を数値で確認できる |
        | **解釈性はやや低い** | 森全体の挙動は人間には見えにくい（ブラックボックス気味） |

        ---
        ### ⚙️ 主なパラメータの意味と直感

        **1️⃣ n_estimators（木の本数）**  
        - 森の中の決定木の数。多いほど安定しますが、計算時間が増えます。  
        - 一般に 100〜300 本で十分（多すぎても大きくは変わらない）。

        **2️⃣ max_depth / min_samples_split / min_samples_leaf**  
        - 各木の「成長のしかた」を制御します。  
        - `max_depth`：木の深さの上限。大きくすると複雑、小さくすると単純。  
        - `min_samples_split`：ノードを分割するために必要なサンプル数。大きいと過学習しにくい。  
        - `min_samples_leaf`：葉に残す最小サンプル数。小さいと細かく分かれるが不安定。

        **3️⃣ max_features（特徴量の上限）**  
        - 各分割で使える特徴量の数。  
        - 少なくすると木ごとに使う特徴がバラバラになり、森が**多様化** → より強いモデルに。

        **4️⃣ bootstrap + oob_score（ブートストラップ & 外れデータ評価）**  
        - 各木を学習させる際に、データを「重複あり」でランダム抽出します（bootstrap）。  
        - 抽出されなかったデータ（約1/3）は「OOB（Out-Of-Bag）」データとして、  
          学習に使わずに性能評価を行う → **追加のテストデータなしで汎化性能を推定可能**。

        ---
        ### 🎓 学びのポイント
        - **単一の木は“極端な意見”を持つが、森全体では“平均的な判断”になる。**  
          → バラつきを抑えた、安定したモデルになる。  
        - **「説明しやすい木」と「予測が強い森」**の違いを理解しよう。  
        - 実務でも「まずCARTで構造を理解 → RandomForestで精度を高める」が基本です。
        """
    )

with st.expander("📊 二変量ヒートマップの解釈（Random Forest）"):
    st.markdown(
        """
        ### 1️⃣ 何を計算しているのか
        ヒートマップでは、次のような手順で **Random Forest の予測値** を可視化しています。

        1. 2つの変数（例：`MedInc`, `HouseAge`）を選ぶ  
        2. それぞれを一定の間隔で区切り（格子状に点を作る）  
        3. 他のすべての変数は **中央値で固定**  
        4. 各点でモデルの予測（`predict()`）を行い、その結果を色で表す  

        → **「他の条件が同じとき、この2変数を変化させたら予測がどう変わるか」**  
        を示した地図のようなものです。

        ---

        ### 2️⃣ 図が意味すること
        - **色の濃淡**：ランダムフォレストが予測した「目的変数（住宅価格など）」の大きさ  
            - 明るい色 → 高い予測値  
            - 暗い色 → 低い予測値  
        - **等高線のような境界**：  
            「どの組み合わせで値が上がる・下がるか」の境界線を示す  

        たとえば：
        | 軸の例 | 読み取り方 |
        |--------|------------|
        | X軸: `MedInc`（地域の所得）<br>Y軸: `AveRooms`（平均部屋数） | 所得と部屋数が多いほど、住宅価格が高くなる傾向 |
        | X軸: `Latitude`（緯度）<br>Y軸: `Longitude`（経度） | 海に近い（南西部）ほど価格が高くなる傾向 |

        ---

        ### 3️⃣ RFモデルの「2次元的反応」を見ている
        このヒートマップは、モデルが学習した **非線形な関係（曲線的な変化）** を  
        「他の変数を固定したうえで」2次元に投影したものです。  

        - CART（決定木）では、分割が直線的でカクカクした境界になる  
        - RandomForestでは、多数の木を平均しているため、  
          **滑らかで現実的な境界**が得られる（平均化の効果）

        ---

        ### 4️⃣ 授業での説明のしかた（イメージ）
        > このヒートマップは、ランダムフォレストが“学んだ世界”を地形図のように描いたものです。  
        > 色の明るいところは“高い予測値（山）”、暗いところは“低い予測値（谷）”。  
        > つまり「この2変数をどう動かすと結果が上がるか・下がるか」を、  
        > モデルが“見える化”しているのです。

        ---
        💡 **まとめ**
        - 2変量ヒートマップは「2つの変数の組み合わせによる予測の変化」を表す。  
        - 他の変数は固定 → 純粋にこの2変数の効果だけを見られる。  
        - 滑らかな色の変化は、ランダムフォレストの**平均化と非線形学習**の結果。
        """
    )


# 6) requirements.txt（コピー用）
REQ_TXT = """
streamlit>=1.37
scikit-learn>=1.2
pandas>=2.1
numpy>=1.26
matplotlib>=3.8
"""
with st.expander("📦 requirements.txt (コピー用)"):
    st.code(REQ_TXT.strip())

