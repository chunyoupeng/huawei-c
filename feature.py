from scipy.stats import skew
import pandas as pd
import numpy as np

def calculate_symmetry(signal):
    # Ensure the signal contains only numeric values, handling non-numeric values and NaN
    signal = pd.to_numeric(signal, errors='coerce')


    # Ensure there are enough data points to calculate symmetry
    if len(signal) == 0:
        return np.nan, np.nan

    # Calculate skewness
    signal_skewness = skew(signal)

    # Calculate center index
    center_index = len(signal) // 2
    left_part = signal[:center_index]
    right_part = signal[center_index:]

    # Compute the means of the left and right parts
    left_mean = np.mean(left_part)
    right_mean = np.mean(right_part)

    # Calculate symmetry measure (difference between left and right means)
    symmetry_measure = left_mean - right_mean

    return signal_skewness, symmetry_measure

def normalize(data):
    min_val = np.min(data)
    max_val = np.max(data)
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data

def extract_frequency_features(signal, fs):
    # 确保信号是 NumPy 数组并转换为 float
    signal = np.asarray(signal, dtype=float)  # 强制转换为 float

    # 检查 fs 合法性
    assert fs > 0, "Sampling frequency fs must be greater than 0."

    # 进行 FFT
    fft_result = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal), d=1/fs)

    # 取前半部分频率和幅值
    half_length = len(signal) // 2
    magnitude = np.abs(fft_result[:half_length])
    frequency = freqs[:half_length]

    # 提取特征
    max_freq = frequency[np.argmax(magnitude)]  # 最大幅值对应的频率
    mean_mag = np.mean(magnitude)  # 幅值均值
    std_mag = np.std(magnitude)    # 幅值标准差
    energy = np.sum(magnitude ** 2)  # 能量

    # 返回特征
    return max_freq, mean_mag, std_mag, energy

def calculate_slope(signal):
    slope = np.zeros_like(signal)  # 创建与信号相同长度的数组
    slope[:-1] = np.diff(signal)  # 前向差分
    return slope

def calculate_duty_cycle(signal, threshold=0):
    # 找到信号大于阈值的部分
    high_time = np.sum(signal > threshold)  # 高电平时间
    total_time = len(signal)  # 总时间 (信号的长度)

    # 计算占空比
    duty_cycle = (high_time / total_time) * 100  # 百分比形式

    return duty_cycle


def calculate_rise_and_fall(signal):
    t = np.arange(len(signal))
    max_value = np.max(signal)
    max_index = np.argmax(signal)

    # 定义阈值（例如最大值的10%和90%）
    threshold_up = max_value * 0.1
    threshold_down = max_value * 0.9

    # 计算上升时间
    rise_start_index = np.where(signal >= threshold_up)[0][0]
    rise_time = t[rise_start_index] - t[0]  # 假设起始时间为t[0]

    # 计算下降时间
    fall_indices = np.where(signal[max_index:] <= threshold_down)[0]
    if fall_indices.size == 0:
        # 如果没有找到下降到阈值以下的点，可以选择返回某个默认值或抛出异常
        fall_time = 0  # 或其他合适的默认值
    else:
        fall_start_index = fall_indices[0] + max_index
        fall_time = t[max_index] - t[fall_start_index] 

    return rise_time, fall_time
def calculate_sudden(signal):
    
    first_derivative = np.gradient(signal)
    # 计算二阶导数
    second_derivative1 = np.gradient(first_derivative)

    # 查找二阶导数的零交点或突变点
    threshold = 0.00005  # 可以根据具体数据调整
    slope_changes = np.where(np.abs(second_derivative1) > threshold)[0]

    # 分离增和降的突变点
    increase_points = slope_changes[second_derivative1[slope_changes] > 0]
    decrease_points = slope_changes[second_derivative1[slope_changes] < 0]
    return len( increase_points ), len(decrease_points)

