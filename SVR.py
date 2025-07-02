import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import time

# --- 1. 参数设置 ---
TIME_STEPS = 24  # 输入步长：使用过去12个点（1小时）
PRED_STEPS = 3  # 输出步长：预测未来3个点（15分钟）
TRAIN_RATIO = 0.8  # 训练集比例


# --- 2. 评估函数 ---
def mean_absolute_percentage_error(y_true, y_pred):
    epsilon = 1e-8
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100


# --- 3. 加载和处理数据 ---
try:
    # index_col=0 将第一列作为DataFrame的索引，不再视为数据列
    df_v = pd.read_csv('V_228.csv', index_col=0)
    flow_data = df_v.values.astype('float32')
    num_stations = flow_data.shape[1]
    print(f"成功加载数据，形状为: {flow_data.shape} (站点数: {num_stations})")
except FileNotFoundError:
    print("错误: V_228.csv 文件未找到。请确保文件与脚本在同一目录下。")
    exit()

# --- 4. 划分训练/测试集并归一化 ---
train_size = int(len(flow_data) * TRAIN_RATIO)
train_data = flow_data[:train_size]
test_data = flow_data[train_size:]

scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train_data)
test_scaled = scaler.transform(test_data)  # 使用训练集的scaler来转换测试集

# --- 5. 为每个站点训练SVR模型并预测 ---
all_predictions = []
all_ground_truth = []
start_time = time.time()

# 循环遍历每一个站点
for i in range(num_stations):
    print(f"正在处理站点 {i + 1}/{num_stations}...")

    # 提取当前站点的单变量时间序列
    station_train_series = train_scaled[:, i]
    station_test_series = test_scaled[:, i]

    # 为当前站点创建滑动窗口数据集
    X_train, y_train = [], []
    for j in range(len(station_train_series) - TIME_STEPS - PRED_STEPS + 1):
        X_train.append(station_train_series[j:(j + TIME_STEPS)])
        y_train.append(station_train_series[(j + TIME_STEPS):(j + TIME_STEPS + PRED_STEPS)])
    X_train, y_train = np.array(X_train), np.array(y_train)

    # 定义SVR模型和MultiOutput包装器
    # 使用RBF核函数来处理非线性关系
    svr = SVR(kernel='rbf')
    multi_output_svr = MultiOutputRegressor(svr)

    # 训练模型
    multi_output_svr.fit(X_train, y_train)

    # 在测试集上进行预测
    X_test, y_test_ground_truth = [], []
    # 注意：测试集样本的构建需要跨越训练集和测试集的边界
    full_series = np.concatenate([station_train_series, station_test_series])

    for j in range(len(train_scaled) - TIME_STEPS - PRED_STEPS + 1, len(full_series) - TIME_STEPS - PRED_STEPS + 1):
        X_test.append(full_series[j:(j + TIME_STEPS)])
        y_test_ground_truth.append(full_series[(j + TIME_STEPS):(j + TIME_STEPS + PRED_STEPS)])
    X_test, y_test_ground_truth = np.array(X_test), np.array(y_test_ground_truth)

    station_predictions = multi_output_svr.predict(X_test)

    all_predictions.append(station_predictions)
    all_ground_truth.append(y_test_ground_truth)

end_time = time.time()
print(f"\n所有站点模型训练和预测完成，耗时: {end_time - start_time:.2f} 秒")

# --- 6. 整理结果并评估 ---
# all_predictions 是一个列表，每个元素是(num_samples, PRED_STEPS)的数组
# 我们需要将其转换为 (num_samples, num_stations, PRED_STEPS)
# 然后reshape以进行反归一化
# [stations, samples, pred_steps] -> [samples, stations, pred_steps]
predictions_reshaped = np.array(all_predictions).transpose(1, 0, 2)
ground_truth_reshaped = np.array(all_ground_truth).transpose(1, 0, 2)

# 为了反归一化，需要将数据变回 (N, num_features) 的2D形状
# 这里我们只反归一化第一个预测步，以简化评估
# 您也可以分别评估每个预测步
pred_step_to_eval = 0
predictions_flat = predictions_reshaped[:, :, pred_step_to_eval]
ground_truth_flat = ground_truth_reshaped[:, :, pred_step_to_eval]

# 创建一个虚拟的2D数组以匹配scaler的输入形状
dummy_pred = np.zeros((predictions_flat.shape[0], num_stations))
dummy_true = np.zeros((ground_truth_flat.shape[0], num_stations))

dummy_pred[:, :] = predictions_flat
dummy_true[:, :] = ground_truth_flat

# 反归一化
predictions_rescaled = scaler.inverse_transform(dummy_pred)
ground_truth_rescaled = scaler.inverse_transform(dummy_true)

# 计算评估指标
mae = mean_absolute_error(ground_truth_rescaled, predictions_rescaled)
rmse = np.sqrt(mean_squared_error(ground_truth_rescaled, predictions_rescaled))
mape = mean_absolute_percentage_error(ground_truth_rescaled, predictions_rescaled)

print("\n--- SVR (支持向量回归) 模型评估结果 ---")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAPE: {mape:.4f} %")