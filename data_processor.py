import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import random
import openpyxl
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from openpyxl import load_workbook
from openpyxl.utils.dataframe import dataframe_to_rows


TRAIN_FILE = r'./data/train_data'
PRED_FILE = r'./data/pred_data'
REPORT_PATH = r'./data/report'
MIN_VALUE = 650
MAX_VALUE = 950
n_components = 20  # 主成分个数
FRESH_TIME = 5  # 水果新鲜的储存时间
STALE_TIME = 30  # 水果不新鲜的储存时间


def get_wave_header(path, skiprows):
    df = pd.read_csv(path, sep='\t', header=None, names=['WaveLength', 'Value'],skiprows=skiprows)
    temp = list(df['WaveLength'])
    wave_header = []
    for item in temp:
        value = float(item)
        wave_header.append(value)
        # if value < MIN_VALUE or value > MAX_VALUE:
        #     continue
        # else:
        #     wave_header.append(value)
    return wave_header


def read_train_data():
    files = os.listdir(TRAIN_FILE)
    path = TRAIN_FILE + '/' + files[0]
    wave_header = get_wave_header(path, 2)
    header = ['storage hour'] + wave_header
    df = pd.DataFrame([], columns=header)
    for file in files:
        # 检查文件是否是txt文件
        if file.endswith(".txt"):
            path = os.path.join(TRAIN_FILE, file)
            # 获取腐败时间
            with open(path, 'r') as f:
                first_line = f.readline()
            temp = first_line.split('\t')
            storage_hour = float(temp[1])
            # 获取波长信息
            data = pd.read_csv(path, sep='\t', header=None, names=['WaveLength', 'Value'], skiprows=2)
            values = list(data['Value'])
            values = [storage_hour] + values
            df.loc[len(df)] = values
    return df


def read_pred_data():
    files = os.listdir(PRED_FILE)
    path = PRED_FILE + '/' + files[0]
    wave_header = get_wave_header(path, 1)
    header = ['file_name'] + wave_header
    df = pd.DataFrame([], columns=header)
    for file in files:
        # 检查文件是否是txt文件
        if file.endswith(".txt"):
            path = os.path.join(PRED_FILE, file)
            # 获取波长信息
            data = pd.read_csv(path, sep='\t', header=None, names=['WaveLength', 'Value'], skiprows=1)
            values = list(data['Value'])
            values = [file] + values
            df.loc[len(df)] = values
    return df


def data_process(data,shuffle=True):
    data_processed = []
    for i in range(len(data)):
        temp = data.iloc[i, 0]  # 获取储存时间或文件名信息
        weaves = np.array(data.columns[1:].tolist())
        wavedata = np.array(data.iloc[i, 1:])
        # 获取波段在规定范围的序号
        indices = [i for i, value in enumerate(weaves)
                   if MIN_VALUE <= value <= MAX_VALUE]
        newdata = wavedata[indices[0]:indices[-1]]
        line_data = np.append([temp], newdata)
        data_processed.append(line_data)
    data_processed = np.array(data_processed)
    if shuffle:
        np.random.shuffle(data_processed)
    x = data_processed[:, 1:]
    y = data_processed[:, 0]
    return x, y


def data_pca(train, pred):
    train_num = len(train)
    pred_num = len(pred)
    data = np.vstack((train, pred))
    pca = PCA(n_components=n_components)
    new_data = pca.fit_transform(data)
    new_train = new_data[:train_num]
    new_pred = new_data[train_num:]
    return new_train, new_pred


def train_model(X, y):
    # model = LinearRegression()
    # model.fit(X, y)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # 初始化线性回归模型
    model = LinearRegression()
    # 训练模型
    model.fit(X_train, y_train)
    # 在测试集上进行预测
    y_pred = model.predict(X_test)

    # 评估模型性能
    score = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    # print("r2_score:", score)
    # print("Mean Squared Error:", mse)
    return score, mse, model


# 模型预测
def predict(model, X_pred, file_names):
    y_pred = model.predict(X_pred)
    header = ['文件名',r'预测腐烂时间(小时)','评价']
    df = pd.DataFrame([], columns=header)
    for index,name in enumerate(file_names):
        if y_pred[index] <= FRESH_TIME:
            evaluate = '新鲜'
        elif FRESH_TIME < y_pred[index] < STALE_TIME:
            evaluate = '不太新鲜'
        else:
            evaluate = '不新鲜'
        df.loc[len(df)] = [name, round(abs(y_pred[index]), 2), evaluate]
    return df


def save_report(name, df):
    path = REPORT_PATH + '/' + name + '.xlsx'
    df.to_excel(path, index=False)  # 不保存索引

    # 打开刚才保存的Excel文件，使用openpyxl调整列宽
    workbook = load_workbook(path)
    worksheet = workbook.active
    # 遍历每一列，根据内容自动调整列宽
    for column_cells in worksheet.columns:
        length = max(len(str(cell.value)) for cell in column_cells)  # 计算最长的字符串长度
        worksheet.column_dimensions[column_cells[0].column_letter].width = length + 2  # 设置列宽，加2是为了留出一些额外空间

    # 保存调整后的Excel文件
    workbook.save(path)


def get_pred_result():
    # 数据读取
    train_data = read_train_data()
    pred_data = read_pred_data()
    # 数据处理
    X_train, y = data_process(train_data,False)
    X_pred, filenames = data_process(pred_data, False)
    # 数据降维
    X_train, X_pred = data_pca(X_train, X_pred)
    # 模型获取
    score, mse, model = train_model(X_train, y)
    # 模型预测
    res = predict(model, X_pred, filenames)
    # 保存预测结果
    save_report('report', res)
    return score, mse, res

#
# score, mse, res = get_pred_result()
# print(res)