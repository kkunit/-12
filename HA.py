#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HA.py

历史平均基线（HA），路径直接在脚本中配置，不用命令行传参。
使用方法：
  1. 修改下面的 ADJ_PATH/SPEED_PATH/TRAIN_RATIO/SAVE_PRED_PATH 四个变量为你自己的地址
  2. 运行 python HA.py 即可
"""

from pathlib import Path
import numpy as np
import pandas as pd

# ====== 配置区 ======
# 邻接矩阵文件（如果不需要可以留空或删除相关代码）
ADJ_PATH       = r"C:\Users\xzf27\Desktop\and\W_228.csv"
# 速度数据文件
SPEED_PATH     = r"C:\Users\xzf27\Desktop\and\V_228.csv"
# 训练集比例
TRAIN_RATIO    = 0.8
# 预测结果保存路径（可改）
SAVE_PRED_PATH = r"C:\Users\xzf27\Desktop\and\pred.csv"
# ====================

def load_adjacency(path: str) -> np.ndarray:
    """读邻接矩阵，当前 HA 不使用它，只是演示。如果不用可以删掉这段。"""
    return pd.read_csv(path, index_col=0).values.astype(np.float32)

def load_speed(path: str) -> pd.DataFrame:
    """
    读速度数据，第一列当索引，解析为 DatetimeIndex，
    并 floor 到最近的 5 分钟，保证所有时间点统一在 00:00/00:05/... 上。
    """
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index)
    # 将 '5T' 改为 '5min' 避免 FutureWarning
    df.index = df.index.floor('5min')
    return df

def build_ha_table(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    根据训练集按“5 分钟槽”取平均，得到 HA 表。
    槽编号 = (小时*60 + 分钟) // 5，范围 0–287。
    返回一个以槽编号为索引的 DataFrame。
    """
    hours = train_df.index.hour
    mins  = train_df.index.minute
    slots = (hours * 60 + mins) // 5
    df2 = train_df.copy()
    df2['_slot'] = slots
    # 只对原有的速度列取平均，_slot 作为 index
    ha = df2.groupby('_slot')[train_df.columns].mean()
    return ha

def ha_forecast(test_df: pd.DataFrame, ha_table: pd.DataFrame) -> pd.DataFrame:
    """
    把 HA 表映射到测试集，返回预测结果 DataFrame。
    用 reindex 对齐槽编号，缺失的槽补 NaN。
    """
    hours = test_df.index.hour
    mins  = test_df.index.minute
    slots = (hours * 60 + mins) // 5
    pred_vals = ha_table.reindex(slots).to_numpy()
    return pd.DataFrame(pred_vals,
                        index=test_df.index,
                        columns=test_df.columns)

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """平均绝对误差."""
    return float(np.mean(np.abs(y_true - y_pred)))

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """均方根误差."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    平均绝对百分比误差。
    为避免除以零，可以在这里对 y_true==0 的地方略过或加一个极小值 eps。
    """
    eps = 1e-6
    return float(np.mean(np.abs((y_true - y_pred) / (y_true + eps)))) * 100

def main():
    print("1. 读取数据 …")
    # 如果不需要邻接矩阵可注释下一行
    _ = load_adjacency(ADJ_PATH)
    speed_df = load_speed(SPEED_PATH)

    print("2. 划分训练 / 测试 …")
    split = int(len(speed_df) * TRAIN_RATIO)
    train_df = speed_df.iloc[:split]
    test_df  = speed_df.iloc[split:]

    print("3. 计算历史平均表 …")
    ha_table = build_ha_table(train_df)

    print("4. 预测 & 评估 …")
    pred_df = ha_forecast(test_df, ha_table)

    y_true = test_df.to_numpy()
    y_pred = pred_df.to_numpy()
    score_mae   = mae(y_true, y_pred)
    score_rmse  = rmse(y_true, y_pred)
    score_mape  = mape(y_true, y_pred)

    print(f"HA 基线评估指标：")
    print(f"  MAE  = {score_mae:.4f}")
    print(f"  RMSE = {score_rmse:.4f}")
    print(f"  MAPE = {score_mape:.4f} %")

    print("5. 保存预测结果 …")
    out_path = Path(SAVE_PRED_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(out_path, index=True)
    print(f"预测结果已保存到 {SAVE_PRED_PATH}")

if __name__ == "__main__":
    main()
