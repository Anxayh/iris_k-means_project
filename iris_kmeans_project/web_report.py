import os
import webbrowser
import pandas as pd
from sklearn.datasets import load_iris

# Iris 数据集翻译成中文
FEATURE_NAME_CN = {
    "sepal length (cm)": "花萼长度 (cm)",
    "sepal width (cm)":  "花萼宽度 (cm)",
    "petal length (cm)": "花瓣长度 (cm)",
    "petal width (cm)":  "花瓣宽度 (cm)",
}

# Iris 三个品种的中文名
TARGET_NAME_CN = {0: "山鸢尾 (Setosa)", 1: "变色鸢尾 (Versicolor)", 2: "维吉尼亚鸢尾 (Virginica)"}


def _df_to_html_table(df, index_name="特征"):
    """
    把 DataFrame 转成带样式的 HTML 表格。
    """
    html = df.to_html(border=0, classes="data-table", header=True, index=True)
    return html


def _build_data_overview(data):
    """
    根据原始数据构建"数据基本信息"区块的 HTML。
    包含：样本数、特征数、数据类型、各特征中文说明。
    """
    n_samples = len(data)
    n_features = data.shape[1]

    rows = ""
    for col in data.columns:
        cn_name = FEATURE_NAME_CN.get(col, col)
        dtype = str(data[col].dtype)
        non_null = int(data[col].notnull().sum())
        rows += f"""
                <tr>
                    <td>{cn_name}</td>
                    <td><code>{col}</code></td>
                    <td>{dtype}</td>
                    <td>{non_null}</td>
                </tr>"""

    html = f"""
            <h2>一、数据基本信息</h2>
            <div class="info-summary">
                <span class="badge">样本总数：{n_samples}</span>
                <span class="badge">特征数量：{n_features}</span>
                <span class="badge">数据来源：scikit-learn Iris 数据集</span>
                <span class="badge">缺失值：0</span>
            </div>
            <table class="data-table">
                <thead>
                    <tr>
                        <th>特征名称（中文）</th>
                        <th>原始列名（英文）</th>
                        <th>数据类型</th>
                        <th>非空数量</th>
                    </tr>
                </thead>
                <tbody>{rows}
                </tbody>
            </table>"""
    return html


def _build_describe(data):
    """
    构建"描述统计"区块的 HTML。
    把 data.describe() 的统计量翻译为中文，列名翻译为中文特征名。
    """
    desc = data.describe()

    # 统计量行名
    desc.index = ["样本数", "均值", "标准差", "最小值", "25%分位数", "中位数", "75%分位数", "最大值"]

    # 特征列名
    desc.columns = [FEATURE_NAME_CN.get(c, c) for c in desc.columns]

    # 保留两位小数
    desc = desc.round(2)

    html = f"""
            <h2>二、描述性统计</h2>
            <p class="section-desc">下表展示了鸢尾花四个特征在 150 个样本上的统计分布情况，包括均值、标准差、四分位数等。</p>
            {_df_to_html_table(desc)}"""
    return html


def _build_missing(data):
    """
    构建"缺失值检查"区块的 HTML。
    """
    miss = data.isnull().sum()
    miss_df = pd.DataFrame({
        "特征名称（中文）": [FEATURE_NAME_CN.get(c, c) for c in miss.index],
        "原始列名（英文）": miss.index,
        "缺失值数量": miss.values,
        "缺失率": [f"{(v / len(data) * 100):.2f}%" for v in miss.values],
    })
    miss_df.index = range(1, len(miss_df) + 1)
    miss_df.index.name = "序号"

    html = f"""
            <h2>三、缺失值检查</h2>
            <p class="section-desc">经检查，所有特征均无缺失值，数据完整，可直接用于建模。</p>
            {_df_to_html_table(miss_df)}"""
    return html


def _build_class_distribution(target):
    """
    构建"真实类别分布"区块的 HTML。
    """
    from collections import Counter
    counts = Counter(target)

    rows = ""
    for label in sorted(counts.keys()):
        cn_name = TARGET_NAME_CN.get(label, f"类别 {label}")
        cnt = counts[label]
        pct = cnt / len(target) * 100
        rows += f"""
                <tr>
                    <td>类别 {label}</td>
                    <td>{cn_name}</td>
                    <td>{cnt}</td>
                    <td>{pct:.1f}%</td>
                </tr>"""

    html = f"""
            <h2>四、真实类别分布</h2>
            <p class="section-desc">Iris 数据集包含三个品种，每个品种各 50 个样本，类别分布均衡。</p>
            <table class="data-table">
                <thead>
                    <tr>
                        <th>类别编号</th>
                        <th>品种名称（中文）</th>
                        <th>样本数</th>
                        <th>占比</th>
                    </tr>
                </thead>
                <tbody>{rows}
                </tbody>
            </table>"""
    return html


def generate_and_open_html(data=None, target=None, best_k=None, score=None,
                           sse_list=None, sil_list=None):
    """
    生成 HTML 可视化报告并在浏览器中打开。

    参数：
        data       原始特征 DataFrame（未标准化），用于展示数据基本信息与描述统计
        target     真实标签，用于展示类别分布
        best_k     最佳簇数
        score      最终轮廓系数
        sse_list   各 k 对应的 SSE 列表
        sil_list   各 k 对应的轮廓系数列表
    """

    # ===== 构建数据信息区块 =====
    data_overview_html = ""
    describe_html = ""
    missing_html = ""
    class_html = ""

    if data is not None:
        data_overview_html = _build_data_overview(data)
        describe_html = _build_describe(data)
        missing_html = _build_missing(data)

    if target is not None:
        class_html = _build_class_distribution(target)

    # ===== 构建聚类结果摘要区块 =====
    result_summary_html = ""
    if best_k is not None and score is not None:
        result_summary_html = f"""
            <h2>五、聚类结果摘要</h2>
            <div class="result-cards">
                <div class="result-card">
                    <div class="result-value">{best_k}</div>
                    <div class="result-label">最佳簇数 K</div>
                </div>
                <div class="result-card">
                    <div class="result-value">{score:.3f}</div>
                    <div class="result-label">轮廓系数</div>
                </div>
                <div class="result-card">
                    <div class="result-value">{len(data) if data is not None else 150}</div>
                    <div class="result-label">聚类样本数</div>
                </div>
                <div class="result-card">
                    <div class="result-value">{'优' if score >= 0.5 else '良' if score >= 0.3 else '一般'}</div>
                    <div class="result-label">聚类效果评级</div>
                </div>
            </div>
            <p class="section-desc">
                通过遍历 K=1~10，结合<b>肘部法则 (SSE)</b> 与<b>轮廓系数</b>双重指标，
                自动确定最佳簇数为 K={best_k}，对应轮廓系数 {score:.3f}。
                {"该系数表明簇内紧凑、簇间分离程度较好，聚类效果理想。" if score >= 0.5 else "该系数表明聚类效果一般，簇间存在一定重叠。"}
            </p>"""

    # ===== 构建 K 值寻优表格 =====
    k_table_html = ""
    if sse_list is not None and sil_list is not None:
        rows = ""
        for i, (sse, sil) in enumerate(zip(sse_list, sil_list), start=1):
            is_best = (best_k is not None and i == best_k)
            highlight = ' class="best-k-row"' if is_best else ""
            sil_str = f"{sil:.3f}" if i >= 2 else "—（k=1 无法计算）"
            best_mark = ' <span class="best-tag">最佳</span>' if is_best else ""
            rows += f"""
                    <tr{highlight}>
                        <td>K = {i}{best_mark}</td>
                        <td>{sse:.2f}</td>
                        <td>{sil_str}</td>
                    </tr>"""

        k_table_html = f"""
            <h2>六、K 值寻优过程</h2>
            <p class="section-desc">下表展示了不同 K 值下的 SSE（误差平方和）与轮廓系数。SSE 越小越好，轮廓系数越接近 1 越好。</p>
            <table class="data-table">
                <thead>
                    <tr>
                        <th>簇数 K</th>
                        <th>SSE（误差平方和）</th>
                        <th>轮廓系数</th>
                    </tr>
                </thead>
                <tbody>{rows}
                </tbody>
            </table>"""

    html_content = f"""
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>鸢尾花 K-Means 聚类分析报告</title>
        <style>
            :root {{
                --primary-color: #2c3e50;
                --accent-color: #3498db;
                --bg-color: #f8f9fa;
                --card-bg: #ffffff;
                --text-color: #333333;
                --table-head-bg: #3498db;
            }}
            body {{
                font-family: 'Segoe UI', 'Microsoft YaHei', Tahoma, sans-serif;
                background-color: var(--bg-color);
                color: var(--text-color);
                margin: 0;
                padding: 40px 20px;
            }}
            .container {{
                max-width: 1100px;
                margin: 0 auto;
                background: var(--card-bg);
                padding: 40px;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
                border-radius: 12px;
            }}
            h1 {{
                text-align: center;
                color: var(--primary-color);
                margin-bottom: 10px;
                font-size: 2.2em;
            }}
            h2 {{
                color: var(--primary-color);
                margin-top: 45px;
                margin-bottom: 20px;
                padding-bottom: 10px;
                border-bottom: 2px solid var(--accent-color);
                font-size: 1.5em;
            }}
            .intro {{
                line-height: 1.8;
                font-size: 1.05em;
                background: #ebf5fb;
                border-left: 5px solid var(--accent-color);
                padding: 20px;
                border-radius: 0 8px 8px 0;
                margin-bottom: 40px;
                color: #555;
            }}
            .section-desc {{
                color: #666;
                margin-bottom: 18px;
                line-height: 1.7;
            }}
            /* 数据表格样式 */
            .data-table {{
                width: 100%;
                border-collapse: collapse;
                margin: 15px 0 25px 0;
                font-size: 0.95em;
                box-shadow: 0 2px 8px rgba(0,0,0,0.04);
                overflow: hidden;
                border-radius: 8px;
            }}
            .data-table thead th {{
                background: var(--table-head-bg);
                color: #fff;
                padding: 12px 15px;
                text-align: left;
                font-weight: 500;
            }}
            .data-table tbody td {{
                padding: 10px 15px;
                border-bottom: 1px solid #eaeaea;
            }}
            .data-table tbody tr:nth-child(even) {{
                background: #f7fbfd;
            }}
            .data-table tbody tr:hover {{
                background: #eaf4fb;
            }}
            .data-table code {{
                background: #f0f0f0;
                padding: 2px 6px;
                border-radius: 3px;
                font-size: 0.9em;
                color: #c0392b;
            }}
            /* 概要徽章 */
            .info-summary {{
                margin: 15px 0 25px 0;
            }}
            .badge {{
                display: inline-block;
                background: #ebf5fb;
                color: var(--accent-color);
                padding: 6px 14px;
                border-radius: 20px;
                margin: 5px 8px 5px 0;
                font-size: 0.9em;
                font-weight: 500;
            }}
            /* 结果卡片 */
            .result-cards {{
                display: flex;
                flex-wrap: wrap;
                gap: 20px;
                margin: 20px 0;
            }}
            .result-card {{
                flex: 1 1 200px;
                background: linear-gradient(135deg, #ebf5fb 0%, #d6eaf8 100%);
                padding: 25px 20px;
                border-radius: 10px;
                text-align: center;
                border: 1px solid #aed6f1;
            }}
            .result-value {{
                font-size: 2.2em;
                font-weight: bold;
                color: var(--accent-color);
            }}
            .result-label {{
                margin-top: 8px;
                color: #555;
                font-size: 0.95em;
            }}
            /* 最佳 K 高亮 */
            .best-k-row {{
                background: #fef9e7 !important;
                font-weight: 500;
            }}
            .best-tag {{
                display: inline-block;
                background: #f39c12;
                color: #fff;
                font-size: 0.75em;
                padding: 2px 8px;
                border-radius: 10px;
                margin-left: 6px;
            }}
            .gallery {{
                display: flex;
                flex-direction: column;
                gap: 40px;
            }}
            .card {{
                text-align: center;
                background: #fff;
            }}
            .card img {{
                max-width: 100%;
                height: auto;
                border: 1px solid #eaeaea;
                border-radius: 8px;
                padding: 8px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.05);
                transition: transform 0.3s ease;
            }}
            .card img:hover {{
                transform: scale(1.02);
            }}
            .card h3 {{
                margin-top: 15px;
                color: var(--primary-color);
                font-weight: 500;
            }}
            .two-col {{
                display: flex;
                flex-wrap: wrap;
                gap: 30px;
            }}
            .two-col .card {{
                flex: 1 1 45%;
            }}
            .small-card {{
                max-width: 500px;
                margin: 0 auto;
            }}
            .footer {{
                text-align: center;
                margin-top: 50px;
                padding-top: 20px;
                border-top: 1px solid #eee;
                color: #888;
                font-size: 0.9em;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>鸢尾花 K-Means 聚类分析报告</h1>
            <div class="intro">
                <strong>项目简介：</strong><br>
                本项目基于 scikit-learn Iris 鸢尾花数据集，利用无监督学习 K-Means 算法进行聚类分析。项目引入了<b>误差平方和 (SSE)</b> 与 <b>轮廓系数 (Silhouette Score)</b> 的双重指标，自适应探索最佳 K 值。为突破 4 维特征的可视化限制，本项目使用 <b>PCA (主成分分析)</b> 算法将高维特征无损降维至二维平面，直观对比了机器学习预测标签与数据真实标签的区别。
            </div>

            {data_overview_html}
            {describe_html}
            {missing_html}
            {class_html}
            {result_summary_html}
            {k_table_html}

            <h2>七、可视化结果</h2>
            <div class="gallery">
                <div class="card">
                    <h3>1. 最佳 K 值探索 (肘部法则与轮廓系数)</h3>
                    <img src="figures/elbow.png" alt="Elbow Method">
                </div>

                <div class="card">
                    <h3>2. 特征成对关系分布图 (Pairplot)</h3>
                    <img src="figures/pairplot.png" alt="Pairplot">
                </div>

                <div class="two-col">
                    <div class="card">
                        <h3>3. K-Means 聚类散点图 (PCA降维后)</h3>
                        <img src="figures/cluster.png" alt="Cluster Result">
                    </div>
                    <div class="card">
                        <h3>4. 真实标签与预测标签对比</h3>
                        <img src="figures/compare.png" alt="Compare True vs Pred">
                    </div>
                </div>

                <div class="card small-card">
                    <h3>5. 最佳 K 值下的模型轮廓系数</h3>
                    <img src="figures/silhouette.png" alt="Silhouette Score">
                </div>
            </div>

            <div class="footer">
                &copy; 软件综合实践 - 基于 K-Means 算法的鸢尾花聚类分析
            </div>
        </div>
    </body>
    </html>
    """

    # 确定生成的 HTML 文件路径
    file_path = os.path.abspath("report.html")

    # 将代码写入 HTML 文件
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"\n✅ 所有图表生成完毕！正在浏览器中为您打开可视化报告...")

    # 调起系统默认浏览器
    webbrowser.open(f"file://{file_path}")


if __name__ == "__main__":
    # 独立运行 web_report.py 时，自动加载数据生成报告
    from analysis import load_data

    data, target = load_data()
    generate_and_open_html(data=data, target=target)
