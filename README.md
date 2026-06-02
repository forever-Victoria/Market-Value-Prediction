# 五大联赛足球运动员身价预测

基于 Transfermarkt 公开数据与传统机器学习方法的**回归**项目，用于体育数据分析类课程实践（球员市场价值预测）。

## 项目简介

- **任务**：根据球员个人属性（年龄、身高、位置、国籍、联赛等）预测其市场身价（`market_value_in_eur`，欧元）
- **联赛范围**：英格兰英超、西班牙西甲、德国德甲、意大利意甲、法国法甲
- **方法**：数据清洗 → 特征工程（含 `appearances.csv` 出场统计）→ 三种回归模型对比 → 特征消融实验
- **环境**：仅需 CPU，不使用 PyTorch / TensorFlow 等深度学习框架

## 目录结构

```
Market Value Prediction/
├── data/                          # 数据集（需自行下载，见下文）
│   ├── players.csv
│   ├── clubs.csv
│   ├── competitions.csv
│   └── ...
├── outputs/                       # 运行脚本后自动生成
│   ├── 01_eda_market_value.png    # 探索性分析图
│   ├── 02_model_metrics_comparison.png
│   ├── 03_feature_importance.png
│   └── model_metrics.csv
├── player_market_value_ml.py      # 主程序（预处理 + 训练 + 评估 + 可视化）
├── report.md                      # 实验报告（方法、理论、结果分析）
├── requirements.txt
└── README.md
```

## 数据获取

本项目使用 [Transfermarkt Datasets](https://github.com/dcaribou/transfermarkt-datasets)（David Caribou）：

| 方式 | 链接 |
|------|------|
| Kaggle | [Football Data from Transfermarkt](https://www.kaggle.com/datasets/davidcariboo/player-scores) |
| GitHub | [dcaribou/transfermarkt-datasets](https://github.com/dcaribou/transfermarkt-datasets) |

下载解压后，将 `players.csv` 等文件放入本项目的 `data/` 目录。

## 环境要求

- Python 3.9+
- Windows / macOS / Linux 均可

## 快速开始

```bash
# 1. 进入项目目录
cd "e:\Market Value Prediction"

# 2. 安装依赖
pip install -r requirements.txt

# 3. 运行完整流水线
python player_market_value_ml.py
```

运行成功后，控制台会输出三模型在测试集上的 **RMSE、MAE、R²** 对比表，图表保存至 `outputs/`。

## 主要配置说明

可在 `player_market_value_ml.py` 顶部修改：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `TOP5_LEAGUE_IDS` | GB1, ES1, L1, IT1, FR1 | 五大联赛 ID |
| `MIN_LAST_SEASON` | 2023 | 筛选活跃球员 |
| `USE_LOG_TARGET` | True | 对身价做 log(1+y) 变换 |
| `TEST_SIZE` | 0.2 | 测试集比例 |
| `RANDOM_STATE` | 42 | 随机种子 |

## 输出说明

| 文件 | 内容 |
|------|------|
| `01_eda_market_value.png` | 身价分布直方图、年龄与身价散点图 |
| `02_model_metrics_comparison.png` | 三模型 RMSE / MAE / R² 柱状对比 |
| `03_feature_importance.png` | 最佳树模型 Top 10 特征重要性 |
| `04_ablation_study.png` | 特征消融实验 R² / RMSE 对比 |
| `model_metrics.csv` | 三模型指标表 |
| `ablation_metrics.csv` | 消融实验指标表 |

## 图表中文显示

脚本会自动检测系统中的中文字体（如 **Microsoft YaHei**）。若图中中文显示为方框，请确认：

1. 系统已安装中文字体；
2. 重新运行 `python player_market_value_ml.py`；
3. 关闭 IDE 中旧的图片预览后重新打开 `outputs/` 下的 PNG。

## 实验报告

完整的方法介绍、理论梳理与结果分析见 **[report.md](./report.md)**。

## 参考文献

- Transfermarkt 数据集：Caribou, D. *transfermarkt-datasets*. GitHub. https://github.com/dcaribou/transfermarkt-datasets  
- scikit-learn 文档：https://scikit-learn.org/stable/

## 许可与声明

数据来源于 Transfermarkt 的公开整理版本，仅供课程学习与研究使用；身价为市场估计值，不代表真实转会成交价。
