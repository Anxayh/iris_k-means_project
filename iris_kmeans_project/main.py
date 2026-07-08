from analysis import load_data
from analysis import scale_data
from analysis import data_info

from clustering import find_best_k
from clustering import run_kmeans

from visualization import draw_cluster
from visualization import draw_compare
from visualization import draw_score
from visualization import draw_elbow
from visualization import draw_pairplot

from web_report import generate_and_open_html

def main():

    # 1. 读取数据
    data, target = load_data()

    # 2. 数据分析
    data_info(data)

    # 3. 数据标准化
    scaled_data, scaler = scale_data(data)

    # 4. 自动寻找最佳 K（肘部法则 + 轮廓系数）
    sse_list, sil_list, best_k = find_best_k(scaled_data)

    # 5. 用最佳 K 做最终聚类
    model, labels, centers, score = run_kmeans(scaled_data, best_k)

    # 6. 绘图
    print("\n正在生成可视化图表，请稍候...")
    draw_elbow(sse_list, sil_list, best_k)     # 肘部法则图
    draw_pairplot(scaled_data, labels)         # 特征关系图
    draw_cluster(scaled_data, labels, centers)  # 聚类结果图
    draw_compare(scaled_data, target, labels)  # 真实与聚类对比图
    draw_score(score)                          # 轮廓系数图

    # 7. 生成 HTML 报告（传入数据与聚类结果，展示数据信息与统计表格）
    generate_and_open_html(
        data=data,
        target=target,
        best_k=best_k,
        score=score,
        sse_list=sse_list,
        sil_list=sil_list,
    )

    print("所有图片已保存到 figures 文件夹。")


if __name__ == "__main__":
    main()
