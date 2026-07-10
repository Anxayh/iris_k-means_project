from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import pandas as pd


def load_data():
    iris = load_iris()

    data = pd.DataFrame(
        iris.data,
        columns=iris.feature_names
    )

    target = iris.target

    return data, target

#特征标准化
def scale_data(data):

    scaler = StandardScaler()

    scaled = scaler.fit_transform(data)

    scaled_df = pd.DataFrame(
        scaled,
        columns=data.columns
    )

    return scaled_df, scaler


def data_info(data):

    print("========== 数据基本信息 ==========")
    print(data.info())

    print("\n========== 描述统计 ==========")
    print(data.describe())

    print("\n========== 缺失值 ==========")
    print(data.isnull().sum())
