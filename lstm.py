import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Reshape
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import time

# --- 1. 参数设置 ---
TIME_STEPS = 24      # 输入步长：使用过去12个点（1小时）
PRED_STEPS = 3       # 输出步长：预测未来3个点（15分钟）
TRAIN_RATIO = 0.8    # 训练集比例
EPOCHS = 50          # 训练轮次
BATCH_SIZE = 32      # 批处理大小

# --- 2. 评估函数 ---
def mean_absolute_percentage_error(y_true, y_pred):
    epsilon = 1e-8
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

# --- 3. 加载和处理数据 ---
try:
    df_v = pd.read_csv('V_228.csv', index_col=0)
    flow_data = df_v.values.astype('float32')
    num_stations = flow_data.shape[1]
    print(f"成功加载数据，形状为: {flow_data.shape}")
except FileNotFoundError:
    print("错误: V_228.csv 文件未找到。请确保文件与脚本在同一目录下。")
    exit()

# --- 4. 划分训练/测试集并归一化 ---
train_size = int(len(flow_data) * TRAIN_RATIO)
train_data = flow_data[:train_size]
test_data = flow_data[train_size:]

scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train_data)
test_scaled = scaler.transform(test_data)

# --- 5. 创建滑动窗口数据集 ---
def create_multivariate_dataset(data, time_steps, pred_steps):
    X, y = [], []
    for i in range(len(data) - time_steps - pred_steps + 1):
        X.append(data[i:(i + time_steps), :])
        # 注意y的形状，需要匹配模型输出
        y.append(data[(i + time_steps):(i + time_steps + pred_steps), :])
    return np.array(X), np.array(y)

# 将训练集和测试集都构造成监督学习样本
X_train, y_train = create_multivariate_dataset(train_scaled, TIME_STEPS, PRED_STEPS)
X_test, y_test = create_multivariate_dataset(test_scaled, TIME_STEPS, PRED_STEPS)

print(f"X_train 形状: {X_train.shape}")
print(f"y_train 形状: {y_train.shape}")
print(f"X_test 形状: {X_test.shape}")
print(f"y_test 形状: {y_test.shape}")


# --- 6. 构建LSTM模型 ---
model = Sequential()
# 输入形状为 (时间步, 特征数/站点数)
model.add(LSTM(64, input_shape=(TIME_STEPS, num_stations)))
# 全连接层输出所有预测步和所有站点的扁平化结果
model.add(Dense(PRED_STEPS * num_stations))
# 将输出结果重塑为 (预测步数, 站点数)
model.add(Reshape((PRED_STEPS, num_stations)))

# 编译模型
model.compile(optimizer=Adam(learning_rate=0.001), loss='mae')
model.summary()

# --- 7. 训练模型 ---
print("\n--- 开始训练LSTM模型 ---")
start_time = time.time()
history = model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1,
    shuffle=True,
    verbose=1
)
end_time = time.time()
print(f"--- LSTM模型训练完成，耗时: {end_time - start_time:.2f} 秒 ---")

# --- 8. 预测与评估 ---
# 在测试集上进行预测
print("\n--- 正在测试集上进行预测 ---")
y_pred_scaled = model.predict(X_test)

# 反归一化
# 为了反归一化，需要将数据变回 (N, num_features) 的2D形状
# y_pred_scaled 和 y_test 的形状都是 (样本数, 预测步长, 站点数)
y_pred_reshaped = y_pred_scaled.reshape(-1, num_stations)
y_test_reshaped = y_test.reshape(-1, num_stations)

y_pred_rescaled = scaler.inverse_transform(y_pred_reshaped)
y_test_rescaled = scaler.inverse_transform(y_test_reshaped)

# 注意：此时 y_pred_rescaled 和 y_test_rescaled 都是2D数组
# 我们直接使用这个2D数组进行评估，不再恢复为3D
print("--- 正在计算评估指标 ---")
mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
rmse = np.sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled))
mape = mean_absolute_percentage_error(y_test_rescaled, y_pred_rescaled)

print("\n--- LSTM 模型评估结果 ---")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAPE: {mape:.4f} %")