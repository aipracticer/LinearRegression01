import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
import numpy as np

class LinearRegression:

    def evaluateModel(self, model, test_data, features, labels):
        """
        计算线性模型的均方差和决定系数

        参数
        ----
        model : LinearRegression, 训练完成的线性模型

        test_data : DataFrame，测试数据

        features : list[str]，特征名列表

        labels : list[str]，标签名列表

        返回
        ----
        error : np.float64，均方差

        score : np.float64，决定系数
        """
        # 均方差(The mean squared error)，均方差越小越好
        error = np.mean(
            (model.predict(test_data[features]) - test_data[labels]) ** 2)
        # R^2: 决定系数(Coefficient of Determination)，R^2 越接近1越好
        score = model.score(test_data[features], test_data[labels])
        return error, score

    def visualizeModel(self, model, data, feature_names, label_names, error, score):
        """
        模型可视化


        参数
        ----
        model: 模型

        data: 数据

        feature_names : 特征名列表

        label_names : 标签名列表

        error: 均方误差

        score: 决定系数
        """
        # Matplotlib中显示中文，需设置特殊字体
        plt.rcParams['font.sans-serif'] = ['SimHei']
        # 创建一个图形框
        fig = plt.figure(figsize=(6, 6), dpi=80)
        # 在图形框里只画一幅图
        ax = fig.add_subplot(111)

        ax.set_title('%s' % "收入与支出关系")

        ax.set_xlabel('收入额')
        ax.set_ylabel('支出额')

        ax.scatter(data[feature_names], data[label_names], color='b')

        if model.intercept_ > 0:
            ax.plot(data[feature_names], model.predict(data[feature_names]), color='r',
                    label='%s: $Y = %.3fX$ + %.3f' \
                          % ("预测值", model.coef_, model.intercept_))

        legend = plt.legend(shadow=True)
        legend.get_frame().set_facecolor('#6F93AE')
        ax.text(0.99, 0.01,
                '%s%.3f\n%s%.3f' \
                % ("均方差 MSE：", error, "决定系数 R^2：", score),
                style='italic', verticalalignment='bottom', horizontalalignment='right',
                transform=ax.transAxes, color='black', fontsize=13)

        plt.show()

    def trainModel(self, train_data, feature_names, label_names):
        """
        利用训练数据，估计模型参数

        参数
        ----
        train_data : DataFrame，训练数据集，包含特征和标签

        feature_names : 特征名列表

        label_names : 标签名列表

        返回
        ----
        model : LinearRegression, 训练好的线性模型
        """
        # 创建一个线性回归模型
        model = linear_model.LinearRegression()
        # 训练模型，估计模型参数
        model.fit(train_data[feature_names], train_data[label_names])
        return model

    def linearModel(self, data, feature_names, label_names, split_ratio):
        """
        线性回归模型建模步骤展示

        参数
        ----
        data : DataFrame，建模数据
        """
        # 划分训练集和测试集
        trainData = data[:int(len(data) * split_ratio)]
        testData = data[int(len(data) * split_ratio):]
        # 产生并训练模型
        model = self.trainModel(trainData, feature_names, label_names)
        # 衡量模型
        error, score = self.evaluateModel(model, testData, feature_names, label_names)
        # 数据可视化模型结果
        self.visualizeModel(model, data, feature_names, label_names, error, score)

    def readData(self, path):
        """
        使用pandas从CSV读取数据

        参数
        ----
        path : 数据的路径
        """
        return pd.read_csv(path)




if __name__ == "__main__":
    prefixPath = os.path.dirname(os.path.abspath(__file__))
    if os.name == "nt":
        dataPath = "%s\\data\\income_vs_consume.csv" % prefixPath
    else:
        dataPath = "%s/data/income_vs_consume.csv" % prefixPath
    feature_names = ["income"]
    label_names = ["consume"]
    split_ratio = 0.75
    lr = LinearRegression()
    data = lr.readData(dataPath)
    lr.linearModel(data, feature_names, label_names, split_ratio)
