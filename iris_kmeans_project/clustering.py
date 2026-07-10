from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


K_RANGE = range(1, 11)  # 候选簇数 1~10


def _detect_elbow(sse_list):
    """
    用最大距离法（Knee 检测）自动定位 SSE 曲线的肘部拐点。

    原理：在 "SSE 关于 K" 的曲线上，画一条连接首尾两点 (K=1, K=10)
    的直线，曲线上离该直线垂直距离最远的点即为"肘部"——即增加簇数
    带来的 SSE 收益开始骤减的拐点，对应数据真实的簇结构断点。

    排除 K=1（单一簇无聚类意义），在 K=2..10 范围内选取距离最大的点。
    """
    # 用整个 K=1..10 的 SSE 构造首尾连线
    ks = list(range(1, len(sse_list) + 1))
    x1, y1 = ks[0], sse_list[0]      # 首点 (1, sse_k1)
    x2, y2 = ks[-1], sse_list[-1]    # 尾点 (10, sse_k10)

    distances = []
    for k, sse in zip(ks, sse_list):
        # 点 (k, sse) 到首尾连线的垂直距离（叉积的绝对值 / 方向向量模长）
        num = abs((y2 - y1) * (k - x1) - (x2 - x1) * (sse - y1))
        den = ((y2 - y1) ** 2 + (x2 - x1) ** 2) ** 0.5
        distances.append(num / den if den > 0 else 0)

    # distances[0] 对应 K=1，跳过；在 K=2..10 中取距离最大者
    candidates = distances[1:]
    elbow_k = candidates.index(max(candidates)) + 2  # +2 还原为实际 K 值
    return elbow_k


def find_best_k(data):
    """
    遍历 k=1..10，计算每个 k 的 SSE（簇内平方和）和轮廓系数。

    K 值选择策略（双指标综合）：
      - 肘部法则（SSE 拐点）：识别数据真实的结构断点，作为主决策依据
      - 轮廓系数：评估聚类质量，作为辅助验证指标

    当两者不一致时（典型如 Iris：肘部指 K=3，轮廓系数偏好 K=2），
    以肘部法则为准——因为轮廓系数在存在强可分子集（如 Setosa 与
    其余两类距离极远）时会偏好偏小的 K，从而掩盖真实的 3 类结构。

    返回：
        sse_list   各 k 对应的 SSE
        sil_list   各 k 对应的轮廓系数（k=1 记 0）
        best_k     肘部法则确定的最佳簇数
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

    # 肘部法则定位拐点（主决策）
    elbow_k = _detect_elbow(sse_list)

    # 轮廓系数最大的 k（辅助验证）
    sil_k = max(
        range(2, len(sil_list) + 1),
        key=lambda i: sil_list[i - 1]
    )

    # 综合决策：以肘部法则为主，轮廓系数为辅
    best_k = elbow_k

    print(f"各 k 的 SSE：{[round(s, 2) for s in sse_list]}")
    print(f"各 k 的轮廓系数：{[round(s, 3) for s in sil_list]}")
    print(f"肘部法则拐点 K = {elbow_k}")
    print(f"轮廓系数最优 K = {sil_k}")
    if elbow_k == sil_k:
        print(f"两指标一致，最佳簇数 k = {best_k}")
    else:
        print(f"两指标不一致，以肘部法则为准，最佳簇数 k = {best_k}")
        print(f"（轮廓系数偏好 K={sil_k}，因存在强可分子集导致分离度被高估）")

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
