# 基于 K-Means 算法实现鸢尾花聚类分析

> 软件综合实践课程项目 — 利用无监督学习 K-Means 算法对 Iris 鸢尾花数据集进行聚类分析，自动寻优最佳簇数，生成可视化 HTML 报告。

## 项目概述

本项目基于著名的 **Iris 鸢尾花数据集**，实现了一套完整的聚类分析流程：

- **数据预处理**：数据加载、标准化处理（Z-score）
- **自动选 K**：结合肘部法则（SSE）与轮廓系数双重指标，遍历 K=1~10 自动确定最佳簇数
- **聚类建模**：使用 scikit-learn K-Means 算法进行无监督聚类
- **多维可视化**：PCA 降维后可视化 + 特征关系图 + 对比分析
- **报告生成**：自动生成包含数据统计、聚类结果、可视化图表的中文 HTML 报告

## 技术栈

| 模块 | 技术 | 版本要求 |
|------|------|----------|
| 机器学习 | scikit-learn | >= 1.3.0 |
| 数据处理 | pandas | >= 2.0.0 |
| 数值计算 | numpy | >= 1.24.0 |
| 可视化 | matplotlib | >= 3.7.0 |
| 可视化 | seaborn | >= 0.12.0 |

## 文件结构

```
iris_kmeans_project/
├── main.py           # 主入口，编排完整分析流程
├── analysis.py       # 数据层：加载、标准化、EDA
├── clustering.py     # 算法层：自动选 K、K-Means 建模
├── visualization.py  # 可视化层：PCA 降维、5 种图表绘制
├── web_report.py     # 报告层：生成中文 HTML 可视化报告
├── requirements.txt  # 依赖声明
├── figures/          # 生成的图表（运行后自动创建）
│   ├── elbow.png     # 肘部法则选 K 图
│   ├── pairplot.png  # 特征关系图
│   ├── cluster.png   # 聚类结果图
│   ├── compare.png   # 真实 vs 预测对比图
│   └── silhouette.png# 轮廓系数图
└── report.html       # 生成的 HTML 报告（运行后自动创建）
```

## 核心功能

### 1. 数据加载与预处理

[analysis.py](file:///h:/programming/iris_kmeans_project/analysis.py) 实现数据加载与标准化：

- `load_data()`：从 scikit-learn 内置 Iris 数据集读取 150 条样本，4 个特征（花萼长度/宽度、花瓣长度/宽度）
- `scale_data()`：使用 `StandardScaler` 做 Z-score 标准化，消除量纲差异（K-Means 基于欧氏距离，量纲不一致会导致大尺度特征主导结果）
- `data_info()`：输出数据类型、描述统计、缺失值检查到终端

### 2. 自动寻找最佳 K

[clustering.py](file:///h:/programming/iris_kmeans_project/clustering.py) 实现智能 K 值寻优：

- `find_best_k()`：遍历 K=1~10，计算每个 K 的 SSE（误差平方和）和轮廓系数，采用**双指标综合选 K**策略：
  - **肘部法则（主决策）**：用最大距离法（Knee 检测）自动定位 SSE 曲线的拐点，识别数据真实的结构断点
  - **轮廓系数（辅验证）**：取值范围 [-1, 1]，越接近 1 表示簇内紧凑、簇间分离
  - **综合决策**：当两指标不一致时（如 Iris 上肘部指 K=3，轮廓系数偏好 K=2），以肘部法则为准——因为轮廓系数在存在强可分子集时会偏好偏小的 K
- `run_kmeans()`：使用最佳 K 进行最终聚类，返回模型、标签、簇心、轮廓系数

### 3. 可视化分析

[visualization.py](file:///h:/programming/iris_kmeans_project/visualization.py) 提供 5 种可视化图表：

| 函数 | 图表 | 描述 |
|------|------|------|
| `draw_elbow()` | elbow.png | 双轴图：左轴 SSE 下降曲线 + 右轴轮廓系数上升曲线，标注最佳 K |
| `draw_pairplot()` | pairplot.png | 特征两两关系图（pairplot），对角线 KDE 分布，非对角线散点图 |
| `draw_cluster()` | cluster.png | PCA 降维后聚类结果散点图，红色 X 标记簇心 |
| `draw_compare()` | compare.png | 左右对比图：左为真实标签，右为聚类结果 |
| `draw_score()` | silhouette.png | 轮廓系数柱状图，标注具体数值 |

**关键设计**：`_get_pca_2d()` 辅助函数确保所有 PCA 降维图共用同一坐标系，保证对比的严谨性。

### 4. HTML 可视化报告

[web_report.py](file:///h:/programming/iris_kmeans_project/web_report.py) 生成中文 HTML 报告，包含 7 个章节：

1. **数据基本信息**：样本数、特征数、各特征中英文对照表、数据类型
2. **描述性统计**：均值、标准差、最小值、四分位数、中位数、最大值（中文翻译）
3. **缺失值检查**：各特征缺失数量与缺失率
4. **真实类别分布**：三个品种（山鸢尾/变色鸢尾/维吉尼亚鸢尾）的样本数与占比
5. **聚类结果摘要**：4 张结果卡片展示最佳 K、轮廓系数、样本数、聚类效果评级
6. **K 值寻优过程**：K=1~10 的 SSE 与轮廓系数表格，最佳 K 行高亮标记
7. **可视化结果**：5 张图表展示

## 快速开始

### 环境配置

```bash
pip install -r requirements.txt
```

### 运行项目

```bash
python main.py
```

运行后将自动完成：
1. 在终端输出数据基本信息、描述统计、缺失值检查、各 K 的 SSE 与轮廓系数、最佳 K、最终轮廓系数
2. 在 `figures/` 目录生成 5 张可视化图表
3. 生成 `report.html` 并在浏览器中自动打开

### 独立生成报告

若仅需生成数据信息报告（不含聚类结果），可单独运行：

```bash
python web_report.py
```

## 运行示例输出

```
========== 数据基本信息 ==========
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 150 entries, 0 to 149
Data columns (total 4 columns):
 #   Column             Non-Null Count  Dtype
---  ------             --------------  -----
 0   sepal length (cm)  150 non-null    float64
 1   sepal width (cm)   150 non-null    float64
 2   petal length (cm)  150 non-null    float64
 3   petal width (cm)   150 non-null    float64

========== 描述统计 ==========
       sepal length (cm)  sepal width (cm)  ...
count         150.000000        150.000000  ...
mean            5.843333          3.057333  ...
std             0.828066          0.435866  ...
...

各 k 的 SSE：[600.0, 222.36, 139.82, 114.09, 90.93, ...]
各 k 的轮廓系数：[0.0, 0.582, 0.46, 0.387, 0.346, ...]
肘部法则拐点 K = 3
轮廓系数最优 K = 2
两指标不一致，以肘部法则为准，最佳簇数 k = 3
最终轮廓系数：0.460

✅ 所有图表生成完毕！正在浏览器中为您打开可视化报告...
```

## 算法原理

### K-Means 算法流程

1. **初始化**：随机选取 K 个簇心
2. **分配**：计算每个样本到各簇心的距离，分配给最近的簇
3. **更新**：计算每个簇的均值，更新簇心位置
4. **迭代**：重复步骤 2-3，直到簇心不再变化或达到最大迭代次数

### 最佳 K 值选择（双指标综合策略）

本项目采用**肘部法则为主、轮廓系数为辅**的综合策略：

1. **肘部法则（主决策）**：用最大距离法（Knee 检测）自动定位 SSE 曲线的拐点。原理是在 K=1..10 的 SSE 曲线上画一条首尾连线，曲线上离该直线垂直距离最远的点即为"肘部"——即增加簇数带来的 SSE 收益开始骤减的结构断点。

2. **轮廓系数（辅验证）**：取值范围 [-1, 1]，越接近 1 表示簇内越紧凑、簇间越分离。但在存在强可分子集（如 Iris 中 Setosa 与其余两类距离极远）时，轮廓系数会偏好偏小的 K。

3. **综合决策**：当两指标不一致时（如 Iris 上肘部指 K=3，轮廓系数偏好 K=2），以肘部法则为准，因为它更能反映数据的真实簇结构。

### PCA 降维

由于 Iris 数据集有 4 个特征，无法直接在二维平面展示聚类结果。使用 PCA 将 4 维特征降维至 2 维，保留最大方差信息，便于可视化分析。

## 项目特色

1. **自动寻优**：无需人工指定 K 值，算法自动探索最佳簇数
2. **双重验证**：结合肘部法则与轮廓系数，结果更可靠
3. **标准化处理**：K-Means 聚类前进行数据标准化，确保各特征权重一致
4. **中文报告**：生成的 HTML 报告包含完整中文翻译，便于撰写课程报告
5. **可复现性**：固定 `random_state=42`，每次运行结果一致
6. **模块化设计**：数据层、算法层、可视化层、报告层分离，职责清晰

## 输出文件说明

| 文件 | 说明 |
|------|------|
| `figures/elbow.png` | 肘部法则与轮廓系数选 K 图 |
| `figures/pairplot.png` | 特征两两关系图 |
| `figures/cluster.png` | K-Means 聚类散点图（PCA 降维后） |
| `figures/compare.png` | 真实标签与聚类结果对比图 |
| `figures/silhouette.png` | 轮廓系数柱状图 |
| `report.html` | 完整中文可视化报告 |

## 注意事项

- 建议使用 Python 3.8+ 版本
- 首次运行需联网下载 scikit-learn 内置数据集
- HTML 报告中的图片引用相对路径，需确保 `figures/` 目录与 `report.html` 在同一目录
- 中文字体优先使用微软雅黑（Microsoft YaHei），如系统未安装会自动降级为其他字体
