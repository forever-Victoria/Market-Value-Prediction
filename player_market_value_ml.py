# -*- coding: utf-8 -*-
"""
五大联赛足球运动员身价预测 —— 传统机器学习完整流程
================================================================================
数据来源：Transfermarkt（data/players.csv 等）
任务类型：回归（预测 market_value_in_eur）
环境约束：仅 CPU，不使用深度学习框架

运行方式（在项目根目录）：
    pip install -r requirements.txt
    python player_market_value_ml.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

# =============================================================================
# 全局配置
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# 五大联赛在 Transfermarkt 中的 competition_id
TOP5_LEAGUE_IDS = ["GB1", "ES1", "L1", "IT1", "FR1"]
TOP5_LEAGUE_NAMES = {
    "GB1": "Premier League",
    "ES1": "La Liga",
    "L1": "Bundesliga",
    "IT1": "Serie A",
    "FR1": "Ligue 1",
}

# 仅保留近年仍在册/有记录的球员，减少退役球员噪声
MIN_LAST_SEASON = 2023
# 从 appearances.csv 聚合近 seasons 的俱乐部比赛表现（显著提升 R²）
USE_APPEARANCE_FEATURES = True
APPEARANCE_SINCE = "2022-07-01"

TARGET_COL = "market_value_in_eur"
# 目标变换：身价分布右偏严重，对数变换可稳定回归残差（报告可写 Box-Cox / log 变换）
USE_LOG_TARGET = True

RANDOM_STATE = 42
TEST_SIZE = 0.2

# 数值特征：身高、国家队、近期俱乐部表现等
NUMERIC_FEATURES = [
    "age",
    "height_in_cm",
    "international_caps",
    "international_goals",
    "last_season",
    "recent_goals",
    "recent_assists",
    "recent_minutes",
    "recent_apps",
    "goals_per90",
    "minutes_per_app",
]

# 近期俱乐部表现字段（来自 appearances.csv 聚合）
APPEARANCE_NUMERIC_FEATURES = [
    "recent_goals",
    "recent_assists",
    "recent_minutes",
    "recent_apps",
    "goals_per90",
    "minutes_per_app",
]

# 分类特征：独热编码；国籍类别过多，后续合并低频为 Other
CATEGORICAL_FEATURES = [
    "position",
    "foot",
    "country_of_citizenship",
    "current_club_domestic_competition_id",
]

# 特征分组（用于方案 B：业务分组消融）
FEATURE_GROUP_BASIC = ["age", "height_in_cm", "position", "foot"]
FEATURE_GROUP_CAREER = [
    "international_caps",
    "international_goals",
    "last_season",
    *APPEARANCE_NUMERIC_FEATURES,
]
FEATURE_GROUP_PERFORMANCE = list(APPEARANCE_NUMERIC_FEATURES)
FEATURE_GROUP_MARKET = [
    "country_of_citizenship",
    "current_club_domestic_competition_id",
]

# 消融实验仅复训最佳模型（梯度提升树），CPU 上秒级完成
ABLATION_MODEL_LABEL = "梯度提升树 (Gradient Boosting)"

# 不参与建模的列（标识、URL、文本等）
DROP_COLUMNS = [
    "player_id",
    "first_name",
    "last_name",
    "name",
    "player_code",
    "url",
    "image_url",
    "agent_name",
    "city_of_birth",
    "country_of_birth",
    "date_of_birth",
    "contract_expiration_date",
    "sub_position",
    "current_club_id",
    "current_national_team_id",
    "current_club_name",
    "highest_market_value_in_eur",  # 历史最高身价 ≈ 强泄漏特征，故剔除
]


def _resolve_chinese_font() -> str | None:
    """
    在系统中查找可用的中文字体。

    Windows 常见：Microsoft YaHei、SimHei。
    注意：不要把 DejaVu Sans 与中文混在同一 sans-serif 列表里，
    否则 matplotlib 可能对中文缺字显示为方框。
    """
    preferred = [
        "Microsoft YaHei",
        "Microsoft YaHei UI",
        "SimHei",
        "SimSun",
        "Source Han Serif SC",
        "Noto Sans CJK SC",
        "PingFang SC",
        "WenQuanYi Micro Hei",
        "Arial Unicode MS",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in preferred:
        if name in available:
            return name
    for font in fm.fontManager.ttflist:
        lower = font.name.lower()
        if "yahei" in lower or "simhei" in lower or "noto sans cjk" in lower:
            return font.name
    return None


def setup_plot_style() -> str | None:
    """统一 matplotlib / seaborn 绘图风格；返回实际使用的中文字体名。"""
    chinese_font = _resolve_chinese_font()
    font_rc = {}
    if chinese_font:
        font_rc = {
            "font.family": "sans-serif",
            "font.sans-serif": [chinese_font],
            "axes.unicode_minus": False,
        }
        matplotlib.rcParams.update(font_rc)
    else:
        matplotlib.rcParams["axes.unicode_minus"] = False
        print("[警告] 未检测到中文字体，图中中文可能无法正常显示。")

    # seaborn 的 set_theme 会重置 rcParams，必须通过 rc= 传入字体配置
    sns.set_theme(
        style="whitegrid",
        palette="husl",
        font_scale=1.05,
        rc=font_rc if font_rc else {"axes.unicode_minus": False},
    )

    if chinese_font:
        matplotlib.rcParams.update(font_rc)

    return chinese_font


def load_and_prepare_raw_data(data_dir: Path) -> pd.DataFrame:
    """
    读取 players.csv，筛选五大联赛、有效身价与合理年龄。

    理论要点：
    - 结构化数据常通过「业务规则过滤」完成第一层清洗（联赛、赛季、目标缺失）。
    - 年龄由 date_of_birth 推导，属于特征工程中的派生变量。
    """
    players_path = data_dir / "players.csv"
    if not players_path.exists():
        raise FileNotFoundError(
            f"未找到 {players_path}，请将 Transfermarkt 的 players.csv 放在 data/ 目录下。"
        )

    df = pd.read_csv(players_path, low_memory=False)
    print(f"[1] 原始球员记录数: {len(df):,}")

    # 五大联赛筛选
    df = df[df["current_club_domestic_competition_id"].isin(TOP5_LEAGUE_IDS)].copy()
    print(f"[2] 五大联赛球员数: {len(df):,}")

    # 近年赛季（活跃球员为主）
    df = df[df["last_season"] >= MIN_LAST_SEASON].copy()
    print(f"[3] last_season >= {MIN_LAST_SEASON} 后: {len(df):,}")

    # 目标变量：有效正身价
    df = df[df[TARGET_COL].notna() & (df[TARGET_COL] > 0)].copy()
    print(f"[4] 有效身价记录数: {len(df):,}")

    # 派生年龄
    df["date_of_birth"] = pd.to_datetime(df["date_of_birth"], errors="coerce")
    reference_date = pd.Timestamp("2025-06-01")
    df["age"] = (
        (reference_date - df["date_of_birth"]).dt.days / 365.25
    ).round(1)
    df = df[df["age"].between(16, 42)].copy()
    print(f"[5] 年龄 ∈ [16, 42] 后: {len(df):,}")

    # 联赛可读名称（用于图表）
    df["league_name"] = df["current_club_domestic_competition_id"].map(TOP5_LEAGUE_NAMES)

    # 身价（百万欧元），便于散点图坐标阅读
    df["market_value_millions"] = df[TARGET_COL] / 1_000_000

    if USE_APPEARANCE_FEATURES:
        df = merge_recent_appearance_stats(df, data_dir)
        print(f"[6] 合并近期出场统计后: {len(df):,}")

    return df.reset_index(drop=True)


def merge_recent_appearance_stats(df: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    """
    从 appearances.csv 聚合球员近期俱乐部比赛表现。

    特征工程要点（报告可写）：
    - 将「逐场记录」聚合为球员级统计，属于典型的 groupby 特征构造；
    - 引入进球、助攻、出场时间等竞技 proxy，弥补 players 表缺少赛季数据的问题；
    - 无出场记录的球员填 0（替补/伤停/新援均合理）。
    """
    appearances_path = data_dir / "appearances.csv"
    if not appearances_path.exists():
        print("[警告] 未找到 appearances.csv，跳过出场特征。")
        for col in APPEARANCE_NUMERIC_FEATURES:
            df[col] = 0.0
        return df

    player_ids = df["player_id"].unique()
    usecols = [
        "player_id",
        "date",
        "goals",
        "assists",
        "minutes_played",
    ]
    apps = pd.read_csv(appearances_path, usecols=usecols)
    apps = apps[apps["player_id"].isin(player_ids)].copy()
    apps["date"] = pd.to_datetime(apps["date"], errors="coerce")
    apps = apps[apps["date"] >= pd.Timestamp(APPEARANCE_SINCE)]

    agg = apps.groupby("player_id", as_index=False).agg(
        recent_goals=("goals", "sum"),
        recent_assists=("assists", "sum"),
        recent_minutes=("minutes_played", "sum"),
        recent_apps=("goals", "count"),
    )

    merged = df.merge(agg, on="player_id", how="left")
    for col in ["recent_goals", "recent_assists", "recent_minutes", "recent_apps"]:
        merged[col] = merged[col].fillna(0)

    merged["goals_per90"] = np.where(
        merged["recent_minutes"] > 0,
        merged["recent_goals"] / merged["recent_minutes"] * 90,
        0.0,
    )
    merged["minutes_per_app"] = np.where(
        merged["recent_apps"] > 0,
        merged["recent_minutes"] / merged["recent_apps"],
        0.0,
    )
    return merged


def reduce_cardinality(
    df: pd.DataFrame, column: str, min_count: int = 40, other_label: str = "Other"
) -> pd.DataFrame:
    """
    高基数分类变量处理：将出现次数 < min_count 的类别合并为 Other。

    理论要点（报告可写）：
    - 独热编码后维度 = 类别数；稀有类别样本不足会导致过拟合。
    - 频率截断 / 目标编码是工业界常用手段，此处用频率截断保持可解释性。
    """
    counts = df[column].fillna("Unknown").astype(str).value_counts()
    keep = counts[counts >= min_count].index
    df = df.copy()
    df[column] = df[column].fillna("Unknown").astype(str)
    df.loc[~df[column].isin(keep), column] = other_label
    return df


def preprocess_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    特征矩阵与目标向量构造（清洗 + 类型处理，尚未标准化/编码）。

    缺失值策略（经典预处理）：
    - 数值：中位数插补（对异常值比均值稳健）
    - 分类：填 Unknown
    - 国家队数据：NaN 视为 0
    """
    work = df.copy()

    work = reduce_cardinality(work, "country_of_citizenship", min_count=40)

    for col in CATEGORICAL_FEATURES:
        work[col] = work[col].fillna("Unknown").astype(str)
    for col in ["foot", "position"]:
        work[col] = work[col].replace("", "Unknown")

    work["international_caps"] = work["international_caps"].fillna(0)
    work["international_goals"] = work["international_goals"].fillna(0)
    work["last_season"] = work["last_season"].fillna(work["last_season"].median())
    work["height_in_cm"] = work["height_in_cm"].fillna(work["height_in_cm"].median())
    for col in APPEARANCE_NUMERIC_FEATURES:
        if col in work.columns:
            work[col] = work[col].fillna(0)

    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = work[feature_cols]
    y = work[TARGET_COL].astype(float)

    return X, y


def build_preprocessor(
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> ColumnTransformer:
    """
    构建 sklearn ColumnTransformer：数值流水线 + 分类流水线。

    理论梳理（报告重点）：
    1) 标准化（StandardScaler）：各特征零均值、单位方差，消除量纲差异；
       对线性回归、基于距离的算法尤为重要。树模型不严格需要，但统一流水线便于对比。
    2) 独热编码（One-Hot Encoding）：将无序分类变量映射为 0/1 向量，避免错误序关系。
    3) Pipeline：将插补、缩放、编码串联，且仅在训练集 fit，防止数据泄漏。

    消融实验时传入不同的 numeric / categorical 列表即可复用同一套预处理逻辑。
    """
    num_cols = numeric_features if numeric_features is not None else NUMERIC_FEATURES
    cat_cols = (
        categorical_features if categorical_features is not None else CATEGORICAL_FEATURES
    )

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    transformers = []
    if num_cols:
        transformers.append(("num", numeric_transformer, num_cols))
    if cat_cols:
        transformers.append(("cat", categorical_transformer, cat_cols))

    if not transformers:
        raise ValueError("至少需要保留一个数值或分类特征。")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def get_feature_names_after_encoding(
    preprocessor: ColumnTransformer,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> list[str]:
    """从 ColumnTransformer 中解析编码后的特征名（用于特征重要性图）。"""
    num_cols = numeric_features if numeric_features is not None else NUMERIC_FEATURES
    cat_cols = (
        categorical_features if categorical_features is not None else CATEGORICAL_FEATURES
    )
    names = list(num_cols)
    if cat_cols and "cat" in preprocessor.named_transformers_:
        ohe: OneHotEncoder = preprocessor.named_transformers_["cat"].named_steps[
            "onehot"
        ]
        names.extend(ohe.get_feature_names_out(cat_cols).tolist())
    return names


def build_gbm_pipeline(
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> Pipeline:
    """构建梯度提升树 Pipeline（消融实验与主实验最佳模型共用结构）。"""
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "model",
                GradientBoostingRegressor(
                    n_estimators=400,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    min_samples_leaf=3,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def build_models() -> dict[str, Pipeline]:
    """
    三种回归模型（均适合结构化表格 + CPU）。

    | 模型 | 原理简述 | 优缺点 |
    |------|----------|--------|
    | 线性回归 | 最小化 MSE，假设线性关系 | 可解释、快；难捕捉非线性 |
    | 随机森林 | Bagging + 决策树，降低方差 | 非线性强、抗过拟合；可输出特征重要性 |
    | 梯度提升树 | Boosting 串行拟合残差 | 精度常更高；训练稍慢 |
    """
    preprocessor = build_preprocessor()

    models = {
        "线性回归 (Linear Regression)": Pipeline(
            steps=[
                ("preprocess", preprocessor),
                (
                    "model",
                    LinearRegression(),
                ),
            ]
        ),
        "随机森林 (Random Forest)": Pipeline(
            steps=[
                ("preprocess", build_preprocessor()),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=300,
                        max_depth=18,
                        min_samples_leaf=3,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "梯度提升树 (Gradient Boosting)": build_gbm_pipeline(),
    }
    return models


def transform_target(y: pd.Series, fit: bool = True) -> tuple[np.ndarray, dict]:
    """目标对数变换 y' = log(1 + y)，缓解身价右偏分布。"""
    y_arr = y.values.astype(float)
    if USE_LOG_TARGET:
        return np.log1p(y_arr), {"use_log": True}
    return y_arr, {"use_log": False}


def inverse_transform_target(y_pred_log: np.ndarray) -> np.ndarray:
    """预测值还原到欧元尺度：exp(y') - 1。"""
    return np.expm1(y_pred_log)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """
    回归评价指标（报告公式可照抄）：

    - RMSE = sqrt(mean((y - ŷ)^2))  ：对大误差敏感，单位与身价相同
    - MAE  = mean(|y - ŷ|)          ：平均绝对误差，更稳健
    - R²   = 1 - SS_res / SS_tot    ：解释方差比例，越接近 1 越好
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"RMSE": rmse, "MAE": mae, "R2": r2}


def train_and_evaluate(
    X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, dict[str, Pipeline], pd.DataFrame, np.ndarray, np.ndarray]:
    """划分训练/测试集，训练三模型并在测试集评估。"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    y_train_t, _ = transform_target(y_train)
    y_test_log, _ = transform_target(y_test)

    models = build_models()
    rows = []

    print("\n" + "=" * 72)
    print("模型训练与测试集评估（身价单位：欧元；RMSE/MAE 同为欧元）")
    print("=" * 72)

    for name, pipe in models.items():
        pipe.fit(X_train, y_train_t)
        pred_log = pipe.predict(X_test)
        pred = inverse_transform_target(pred_log)
        y_true = y_test.values

        metrics = compute_metrics(y_true, pred)
        metrics["模型"] = name
        rows.append(metrics)

        print(f"\n>>> {name}")
        print(f"    RMSE : {metrics['RMSE']:,.0f} 欧元")
        print(f"    MAE  : {metrics['MAE']:,.0f} 欧元")
        print(f"    R2   : {metrics['R2']:.4f}")

    results_df = pd.DataFrame(rows)[["模型", "RMSE", "MAE", "R2"]]

    print("\n" + "-" * 72)
    print("模型对比汇总表")
    print("-" * 72)
    display_df = results_df.copy()
    display_df["RMSE"] = display_df["RMSE"].apply(lambda v: f"{v:,.0f}")
    display_df["MAE"] = display_df["MAE"].apply(lambda v: f"{v:,.0f}")
    display_df["R2"] = display_df["R2"].map(lambda v: f"{v:.4f}")
    print(display_df.to_string(index=False))
    print("-" * 72)

    return results_df, models, X_test, y_test.values, y_test_log


def plot_eda(df: pd.DataFrame, output_dir: Path) -> None:
    """数据探索可视化：身价分布 + 年龄与身价关系。"""
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 直方图：身价（百万欧）右偏
    sns.histplot(
        df["market_value_millions"],
        bins=50,
        kde=True,
        ax=axes[0],
        color="#2E86AB",
        edgecolor="white",
    )
    axes[0].set_xlabel("身价（百万欧元）")
    axes[0].set_ylabel("球员人数")
    axes[0].set_title("五大联赛球员身价分布")
    axes[0].set_xlim(0, df["market_value_millions"].quantile(0.98))

    # 散点图：年龄 vs 身价（抽样避免过密）
    sample = df.sample(n=min(2500, len(df)), random_state=RANDOM_STATE)
    sns.scatterplot(
        data=sample,
        x="age",
        y="market_value_millions",
        hue="league_name",
        alpha=0.55,
        s=35,
        ax=axes[1],
    )
    axes[1].set_xlabel("年龄")
    axes[1].set_ylabel("身价（百万欧元）")
    axes[1].set_title("年龄与身价关系（按联赛着色）")
    axes[1].set_ylim(0, sample["market_value_millions"].quantile(0.98))
    axes[1].legend(title="联赛", fontsize=8, loc="upper right")

    plt.tight_layout()
    path = output_dir / "01_eda_market_value.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[图] 已保存: {path}")


def plot_metrics_comparison(results_df: pd.DataFrame, output_dir: Path) -> None:
    """三模型 RMSE / MAE / R² 柱状对比图。"""
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_df = results_df.copy()
    # 缩短横轴标签
    plot_df["模型简称"] = plot_df["模型"].str.replace(r"\s*\(.*\)", "", regex=True)

    metrics = ["RMSE", "MAE", "R2"]
    titles = ["RMSE（越低越好）", "MAE（越低越好）", "R²（越高越好）"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, metric, title in zip(axes, metrics, titles):
        colors = sns.color_palette("Set2", n_colors=len(plot_df))
        bars = ax.bar(plot_df["模型简称"], plot_df[metric], color=colors, edgecolor="#333")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=15)

        if metric == "R2":
            ax.set_ylim(0, min(1.0, plot_df[metric].max() * 1.15 + 0.05))
        else:
            ax.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x:.0f}")
            )

        for bar, val in zip(bars, plot_df[metric]):
            if metric == "R2":
                label = f"{val:.3f}"
            else:
                label = f"{val/1e6:.2f}M"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                label,
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.suptitle("回归模型测试集指标对比", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = output_dir / "02_model_metrics_comparison.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[图] 已保存: {path}")


def plot_feature_importance(
    models: dict[str, Pipeline],
    results_df: pd.DataFrame,
    X_train_sample: pd.DataFrame,
    output_dir: Path,
    top_k: int = 10,
) -> str:
    """
    从测试集 R² 最高的树模型中提取特征重要性（Top-K）。

    理论要点：
    - 随机森林：节点分裂带来的 MSE 下降量（Gini/方差减少）累计为 importance。
    - 梯度提升：同理，反映各特征对残差拟合的贡献。
    - 线性模型可改用 |系数|，但本函数优先树模型以便汇报。
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    best_row = results_df.loc[results_df["R2"].idxmax()]
    best_name = best_row["模型"]

    # 在两个树模型中选 R² 更高者
    tree_names = [
        "随机森林 (Random Forest)",
        "梯度提升树 (Gradient Boosting)",
    ]
    tree_results = results_df[results_df["模型"].isin(tree_names)]
    best_tree_name = tree_results.loc[tree_results["R2"].idxmax(), "模型"]

    pipe = models[best_tree_name]
    preprocessor: ColumnTransformer = pipe.named_steps["preprocess"]
    model = pipe.named_steps["model"]

    # 在训练样本上 fit 变换器以获取特征名（pipeline 已 fit 过，直接 transform）
    feature_names = get_feature_names_after_encoding(preprocessor)

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        raise ValueError("所选模型不支持 feature_importances_")

    imp_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(top_k)
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=imp_df,
        y="feature",
        x="importance",
        hue="feature",
        palette="viridis",
        legend=False,
        ax=ax,
    )
    ax.set_xlabel("特征重要性")
    ax.set_ylabel("特征")
    ax.set_title(f"Top {top_k} 特征重要性 —— {best_tree_name}")
    plt.tight_layout()
    path = output_dir / "03_feature_importance.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[图] 已保存: {path}（最佳树模型: {best_tree_name}，全局最佳: {best_name}）")
    return best_tree_name


def get_raw_feature_importance(
    pipe: Pipeline,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> pd.Series:
    """
    将独热编码后的 feature_importances_ 聚合回原始特征列。

    例如 position_Attack + position_Defender 的权重合并为 position，
    便于消融实验按「业务字段」剔除特征。
    """
    preprocessor: ColumnTransformer = pipe.named_steps["preprocess"]
    model = pipe.named_steps["model"]
    num_cols = numeric_features if numeric_features is not None else NUMERIC_FEATURES
    cat_cols = (
        categorical_features if categorical_features is not None else CATEGORICAL_FEATURES
    )
    feature_names = get_feature_names_after_encoding(
        preprocessor, num_cols, cat_cols
    )
    importances = model.feature_importances_
    raw_scores: dict[str, float] = {}

    for fname, score in zip(feature_names, importances):
        raw_name = fname
        for cat_col in cat_cols:
            if fname.startswith(f"{cat_col}_"):
                raw_name = cat_col
                break
        raw_scores[raw_name] = raw_scores.get(raw_name, 0.0) + float(score)

    return pd.Series(raw_scores).sort_values(ascending=False)


def _fit_eval_gbm_on_feature_subset(
    X: pd.DataFrame,
    y: pd.Series,
    numeric_features: list[str],
    categorical_features: list[str],
) -> dict[str, float]:
    """在指定特征子集上训练梯度提升树，并在同一随机划分的测试集上评估。"""
    feature_cols = numeric_features + categorical_features
    X_sub = X[feature_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X_sub, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    y_train_t, _ = transform_target(y_train)

    pipe = build_gbm_pipeline(numeric_features, categorical_features)
    pipe.fit(X_train, y_train_t)
    pred = inverse_transform_target(pipe.predict(X_test))
    return compute_metrics(y_test.values, pred)


def run_ablation_study(
    X: pd.DataFrame,
    y: pd.Series,
    models: dict[str, Pipeline],
) -> pd.DataFrame:
    """
    特征消融实验（Ablation Study）。

    理论要点（报告可写）：
    - 消融实验通过「去掉某组特征后性能下降多少」验证特征贡献，而不仅是相关性。
    - 与特征重要性图形成闭环：重要性高 + 去掉后 R² 明显下降 → 因果式论证更充分。

    方案 A：剔除完整模型中 Top-3 原始特征后重训。
    方案 B：按业务分组，仅保留基础信息 / 生涯表现 / 市场背景其中一类。
    """
    baseline_pipe = models[ABLATION_MODEL_LABEL]
    raw_importance = get_raw_feature_importance(baseline_pipe)
    top3_remove = raw_importance.head(3).index.tolist()

    num_all = list(NUMERIC_FEATURES)
    cat_all = list(CATEGORICAL_FEATURES)

    scenarios: list[tuple[str, list[str], list[str], str]] = [
        (
            "完整特征（基准）",
            num_all,
            cat_all,
            "基准：全部特征",
        ),
        (
            f"剔除 Top-3 特征 ({', '.join(top3_remove)})",
            [c for c in num_all if c not in top3_remove],
            [c for c in cat_all if c not in top3_remove],
            f"方案 A：去掉重要性最高的 3 个原始字段 {top3_remove}",
        ),
        (
            "仅基础信息",
            [c for c in FEATURE_GROUP_BASIC if c in num_all],
            [c for c in FEATURE_GROUP_BASIC if c in cat_all],
            "方案 B：年龄、身高、位置、惯用脚",
        ),
        (
            "仅生涯/竞技 proxy",
            [c for c in FEATURE_GROUP_CAREER if c in num_all],
            [c for c in FEATURE_GROUP_CAREER if c in cat_all],
            "方案 B：国家队、最近赛季、近期俱乐部进球/助攻/出场",
        ),
        (
            "仅近期俱乐部表现",
            [c for c in FEATURE_GROUP_PERFORMANCE if c in num_all],
            [],
            "方案 B：仅 appearances 聚合的进球/助攻/出场时间",
        ),
        (
            "仅市场/背景",
            [c for c in FEATURE_GROUP_MARKET if c in num_all],
            [c for c in FEATURE_GROUP_MARKET if c in cat_all],
            "方案 B：国籍、所属联赛",
        ),
    ]

    rows = []
    baseline_r2 = None
    baseline_rmse = None

    print("\n" + "=" * 72)
    print("特征消融实验（梯度提升树，测试集评估）")
    print("=" * 72)
    print(f"完整模型 Top-3 原始特征（将用于方案 A 剔除）: {top3_remove}")

    for scenario_name, num_feats, cat_feats, note in scenarios:
        if not num_feats and not cat_feats:
            continue

        metrics = _fit_eval_gbm_on_feature_subset(X, y, num_feats, cat_feats)
        row = {
            "消融场景": scenario_name,
            "说明": note,
            "特征数_编码前": len(num_feats) + len(cat_feats),
            "RMSE": metrics["RMSE"],
            "MAE": metrics["MAE"],
            "R2": metrics["R2"],
        }

        if baseline_r2 is None:
            baseline_r2 = metrics["R2"]
            baseline_rmse = metrics["RMSE"]
            row["R2_相对基准变化"] = 0.0
            row["RMSE_相对基准升幅_pct"] = 0.0
        else:
            row["R2_相对基准变化"] = metrics["R2"] - baseline_r2
            if baseline_rmse > 0:
                row["RMSE_相对基准升幅_pct"] = (
                    (metrics["RMSE"] - baseline_rmse) / baseline_rmse * 100
                )
            else:
                row["RMSE_相对基准升幅_pct"] = 0.0

        rows.append(row)
        print(
            f"\n>>> {scenario_name}\n"
            f"    R2   : {metrics['R2']:.4f}  "
            f"(Δ {row['R2_相对基准变化']:+.4f})\n"
            f"    RMSE : {metrics['RMSE']:,.0f}  "
            f"(+{row['RMSE_相对基准升幅_pct']:.1f}%)"
        )

    ablation_df = pd.DataFrame(rows)
    print("\n" + "-" * 72)
    print("消融实验汇总")
    print("-" * 72)
    show = ablation_df[
        ["消融场景", "R2", "RMSE", "R2_相对基准变化", "RMSE_相对基准升幅_pct"]
    ].copy()
    show["R2"] = show["R2"].map(lambda v: f"{v:.4f}")
    show["RMSE"] = show["RMSE"].map(lambda v: f"{v:,.0f}")
    show["R2_相对基准变化"] = show["R2_相对基准变化"].map(lambda v: f"{v:+.4f}")
    show["RMSE_相对基准升幅_pct"] = show["RMSE_相对基准升幅_pct"].map(
        lambda v: f"{v:+.1f}%"
    )
    print(show.to_string(index=False))
    print("-" * 72)

    return ablation_df


def plot_ablation_study(ablation_df: pd.DataFrame, output_dir: Path) -> None:
    """绘制消融实验 R² 对比与相对基准的 RMSE 升幅。"""
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_df = ablation_df.copy()
    plot_df["场景简称"] = plot_df["消融场景"].str.replace(
        r"（基准）", "", regex=False
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    colors = sns.color_palette("Set2", n_colors=len(plot_df))
    bars_r2 = axes[0].bar(
        range(len(plot_df)),
        plot_df["R2"],
        color=colors,
        edgecolor="#333",
    )
    axes[0].set_xticks(range(len(plot_df)))
    axes[0].set_xticklabels(plot_df["场景简称"], rotation=20, ha="right", fontsize=9)
    axes[0].set_ylabel("R²")
    axes[0].set_title("消融实验：测试集 R² 对比（越高越好）")
    axes[0].set_ylim(0, max(0.6, plot_df["R2"].max() * 1.12))

    for bar, val in zip(bars_r2, plot_df["R2"]):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    rmse_pct = plot_df["RMSE_相对基准升幅_pct"]
    bars_rmse = axes[1].bar(
        range(len(plot_df)),
        rmse_pct,
        color=colors,
        edgecolor="#333",
    )
    axes[1].set_xticks(range(len(plot_df)))
    axes[1].set_xticklabels(plot_df["场景简称"], rotation=20, ha="right", fontsize=9)
    axes[1].set_ylabel("RMSE 相对基准升幅 (%)")
    axes[1].set_title("消融实验：RMSE 恶化幅度（相对完整特征）")
    axes[1].axhline(0, color="#666", linewidth=0.8)

    for bar, val in zip(bars_rmse, rmse_pct):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            max(bar.get_height(), 0) + 1,
            f"{val:+.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.suptitle(
        "特征消融实验 —— 梯度提升树（验证核心特征贡献）",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    path = output_dir / "04_ablation_study.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[图] 已保存: {path}")


def main() -> None:
    """主流程：加载 → 预处理 → 训练评估 → 可视化。"""
    font_used = setup_plot_style()
    if font_used:
        print(f"[字体] 图表中文将使用: {font_used}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("五大联赛足球运动员身价预测 —— 机器学习流水线")
    print("=" * 72)

    df = load_and_prepare_raw_data(DATA_DIR)
    plot_eda(df, OUTPUT_DIR)

    X, y = preprocess_features(df)
    print(f"\n特征维度（编码前）: {X.shape[1]} 列, 样本数: {len(X):,}")

    results_df, models, X_test, y_test, _ = train_and_evaluate(X, y)
    plot_metrics_comparison(results_df, OUTPUT_DIR)

    # 用训练集子集画特征重要性（完整训练集已在 pipeline 内 fit）
    X_train, _, _, _ = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    plot_feature_importance(models, results_df, X_train, OUTPUT_DIR)

    # 特征消融实验：与特征重要性形成闭环
    ablation_df = run_ablation_study(X, y, models)
    plot_ablation_study(ablation_df, OUTPUT_DIR)
    ablation_path = OUTPUT_DIR / "ablation_metrics.csv"
    ablation_df.to_csv(ablation_path, index=False, encoding="utf-8-sig")

    # 保存指标 CSV 供报告引用
    metrics_path = OUTPUT_DIR / "model_metrics.csv"
    results_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    print(f"\n[完成] 指标表已保存: {metrics_path}")
    print(f"[完成] 消融实验表已保存: {ablation_path}")
    print(f"[完成] 所有图表目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
