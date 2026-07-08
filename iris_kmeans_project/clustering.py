from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


K_RANGE = range(1, 11)  # 候选簇数 1~10


def find_best_k(data):
    """
    遍历 k=1..10，计算每个 k 的 SSE（簇内平方和）和轮廓系数。
    - SSE 用于肘部法则：SSE 随 k 增大单调下降，下降变缓处即"肘部"。
    - 轮廓系数用于度量聚类质量，越大越好（k=1 时无定义，记为 0）。
    返回：
        sse_list   各 k 对应的 SSE
        sil_list   各 k 对应的轮廓系数（k=1 记 0）
        best_k     轮廓系数最大的 k
    """
    sse_list = []
    sil_list = []

    for k in K_RANGE:
        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        model.fit(data)

        # inertia_ 即簇内平方和 SSE
        sse_list.append(model.inertia_)

        if k >= 2:
            score = silhouette_score(data, model.labels_)
        else:
            score = 0.0  # k=1 无法计算轮廓系数

        sil_list.append(score)

    # 选择轮廓系数最大的 k（跳过 k=1）
    best_k = max(
        range(2, len(sil_list) + 1),
        key=lambda i: sil_list[i - 1]
    )

    print(f"各 k 的 SSE：{[round(s, 2) for s in sse_list]}")
    print(f"各 k 的轮廓系数：{[round(s, 3) for s in sil_list]}")
    print(f"最佳簇数 k = {best_k}")

    return sse_list, sil_list, best_k


def run_kmeans(data, k):
    """
    用指定的 k 进行最终聚类。
    返回模型、标签、簇心、轮廓系数。
    """
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(data)

    labels = model.labels_

    centers = model.cluster_centers_

    score = silhouette_score(data, labels)

    print(f"最终轮廓系数：{score:.3f}")

    return model, labels, centers, score
