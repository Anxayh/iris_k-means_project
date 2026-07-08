import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

# 中文字体设置 
# 优先使用微软雅黑，SimHei备用
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

# 创建图片保存目录
os.makedirs("figures", exist_ok=True)


def _get_pca_2d(data, centers=None):
    """
    只对数据 fit 一次 PCA，保证所有图共用同一套坐标系。
    返回 (data_2d, centers_2d_or_None, pca_model)
    """
    pca = PCA(n_components=2)
    data_2d = pca.fit_transform(data)
    if centers is not None:
        if not isinstance(centers, pd.DataFrame):
            centers = pd.DataFrame(centers, columns=data.columns)
        centers_2d = pca.transform(centers)
    else:
        centers_2d = None
    return data_2d, centers_2d, pca


def draw_cluster(data, labels, centers):
    """
    绘制 K-Means 聚类结果
    保存：figures/cluster.png
    """
    data_2d, centers_2d, _ = _get_pca_2d(data, centers)

    plt.figure(figsize=(8, 6))

    plt.scatter(
        data_2d[:, 0],
        data_2d[:, 1],
        c=labels,
        cmap="viridis",
        s=50
    )

    plt.scatter(
        centers_2d[:, 0],
        centers_2d[:, 1],
        marker="X",
        s=250,
        c="red",
        label="簇心"
    )

    plt.title("K-Means 聚类结果")
    plt.xlabel("主成分 1")
    plt.ylabel("主成分 2")
    plt.legend()

    plt.tight_layout()
    plt.savefig("figures/cluster.png", dpi=300)
    plt.close()


def draw_compare(data, true_labels, pred_labels):
    """
    绘制真实分类与聚类结果对比图
    保存：figures/compare.png
    """
    # centers=None，只用数据 fit PCA，与 draw_cluster 共用同一坐标系
    data_2d, _, _ = _get_pca_2d(data, None)

    plt.figure(figsize=(12, 5))

    # 左：真实标签
    plt.subplot(1, 2, 1)
    plt.scatter(
        data_2d[:, 0],
        data_2d[:, 1],
        c=true_labels,
        cmap="viridis",
        s=50
    )
    plt.title("真实标签")
    plt.xlabel("主成分 1")
    plt.ylabel("主成分 2")

    # 右：聚类结果
    plt.subplot(1, 2, 2)
    plt.scatter(
        data_2d[:, 0],
        data_2d[:, 1],
        c=pred_labels,
        cmap="viridis",
        s=50
    )
    plt.title("聚类结果")
    plt.xlabel("主成分 1")
    plt.ylabel("主成分 2")

    plt.tight_layout()
    plt.savefig("figures/compare.png", dpi=300)
    plt.close()


def draw_score(score):
    """
    绘制轮廓系数柱状图
    保存：figures/silhouette.png
    """
    plt.figure(figsize=(5, 5))

    plt.bar(
        ["轮廓系数"],
        [score],
        width=0.4
    )

    plt.ylim(0, 1)

    plt.text(
        0,
        score + 0.03,
        f"{score:.3f}",
        ha="center",
        fontsize=12
    )

    plt.title("轮廓系数")

    plt.tight_layout()
    plt.savefig("figures/silhouette.png", dpi=300)
    plt.close()


def draw_elbow(sse_list, sil_list, best_k):
    """
    绘制肘部法则图：左轴 SSE（下降曲线），右轴轮廓系数（上升曲线）。
    标注最佳k。
    保存：figures/elbow.png
    """
    ks = list(range(1, len(sse_list) + 1))

    fig, ax1 = plt.subplots(figsize=(9, 6))

    # 左轴：SSE
    color1 = "tab:blue"
    ax1.set_xlabel("簇数 k")
    ax1.set_ylabel("SSE（簇内平方和）", color=color1)
    ax1.plot(ks, sse_list, "o-", color=color1, label="SSE")
    ax1.tick_params(axis="y", labelcolor=color1)

    # 右轴：轮廓系数
    ax2 = ax1.twinx()
    color2 = "tab:red"
    ax2.set_ylabel("轮廓系数", color=color2)
    ax2.plot(ks, sil_list, "s--", color=color2, label="轮廓系数")
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_ylim(0, 1)

    # 标注最佳 k
    ax1.axvline(x=best_k, color="green", linestyle=":", label=f"最佳 k={best_k}")

    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    plt.title("肘部法则与轮廓系数选 K")
    fig.tight_layout()
    plt.savefig("figures/elbow.png", dpi=300)
    plt.close()


def draw_pairplot(data, labels=None):
    """
    绘制特征两两关系图（pairplot），用颜色区分簇。
    保存：figures/pairplot.png
    """
    df = data.copy()

    if labels is not None:
        df["簇"] = labels
        hue_col = "簇"
    else:
        hue_col = None

    g = sns.pairplot(
        df,
        hue=hue_col,
        diag_kind="kde",
        plot_kws={"s": 30, "alpha": 0.8}
    )

    g.fig.suptitle("鸢尾花特征关系图", y=1.02, fontsize=14)

    g.savefig("figures/pairplot.png", dpi=300)
    plt.close()
