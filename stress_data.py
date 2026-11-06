from pyedflib import highlevel
import pandas as pd
import numpy as np
import os
# from google.colab import drive
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import simps
from scipy import signal
from scipy.signal import cheby2, butter, filtfilt
# from dtaidistance import dtw
from scipy.ndimage import gaussian_filter
from scipy.signal import resample, convolve
from scipy.signal import find_peaks
from scipy.interpolate import CubicSpline, UnivariateSpline, interp1d
from scipy.signal import welch
from scipy.fft import rfft, rfftfreq
from scipy.stats import linregress, gaussian_kde
from scipy.optimize import curve_fit
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_recall_curve, auc
from xgboost import XGBClassifier

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

def segment_process(data, col, segments, sampling_rate):
  data_length = data.shape[0]
  sensor_per_minute = sampling_rate * 60
  during_points = 0
  labels = np.zeros(data_length)

  future_segments = [seg for seg in segments if seg[1] is not None][-3:]
  future_points = sum(sensor_per_minute * seg[1] for seg in future_segments)

  for name, duration in segments:
    if duration is not None:
      segment_points = sensor_per_minute * duration
    else:
      segment_points = data_length - during_points - future_points

    end_idx = during_points + segment_points
    labels[during_points:end_idx] = 0 if 'Rest' in name else 1
    during_points = end_idx

  data['labels'] = labels
  data[col] = pd.to_numeric(data[col], errors='coerce')
  data[col] = data[col].astype(float)
  return data

  # 체비셰프 필터
# 저주파 필터링(Low-pass filtering) - 버터워스 필터
def chevyfilter_process(data, col, lowcut = 0.5, highcut = 5.0, fs = 64, order = 4, rp = 20):

  data[col] = pd.to_numeric(data[col], errors = 'coerce')
  data[col] = data[col].astype(float)

  # Chebyshev filter II
  nyquist = 0.5 * fs
  low = lowcut / nyquist
  high = highcut / nyquist
  b, a = cheby2(order, rp, [low, high], btype = 'bandpass', analog = False)
  signal_filtered = filtfilt(b, a, data[col].values)

  # 피크 감도를 높이기 위해 신호를 정규화.
  # 피크를 더 두두러지게 하기 위함.
  signal_filtered = (signal_filtered - signal_filtered.mean()) / signal_filtered.std()
  # signal_filtered = np.gradient(signal_filtered)

  data[col] = signal_filtered
  return data

def interpolate_missing_values(data, col, fs = 64):
  data[col] = pd.to_numeric(data[col], errors = 'coerce')
  t = np.arange(len(data[col])) / fs
  valid_indices = ~np.isnan(data[col]) # valid_indices = ~data[col].isnull()
  t_valid = t[valid_indices]
  data_valid = data[col][valid_indices]

  # spline = CubicSpline(t_valid, data_valid)
  interpolation = interp1d(t_valid, data_valid, kind ='linear', fill_value = 'extrapolate')

  interpolated = data.copy()
  interpolated[col] = interpolated[col].fillna(method = 'ffill').fillna(method = 'bfill')
  # interpolated.loc[np.isnan(data[col]), col] = interpolation(t[np.isnan(data[col])])

  return interpolated

def peak_detection(data, col, min_distance = 0.4, min_height = 0.4, fs = 64):
  distance = int(fs * min_distance)
  # if len(data[col]) > 1000:
  #   distance = int(distance * 1.5)
  peaks, _ = find_peaks(data[col], height = min_height, distance = distance) # , width=(None, 50), prominence=0.1
  return peaks

# def peak_detection(data, col, min_distance = 0.4, min_height = 0.4, fs = 100):
#     peaks, _ = find_peaks(data[col],height = 0)
#     return peaks

def calculate_peak_to_peak_intervals(data, peaks, fs = 64):
  peak_times = peaks / fs * 1000
  intervals = np.diff(peak_times)
  valid_indices = (intervals >= 600) & (intervals <= 1200)
  valid_intervals = intervals[valid_indices]
  valid_peaks = peaks[1:][valid_indices]
  return valid_peaks, valid_intervals

def adjust_intervals(data, valid_peaks, valid_intervals):

  if valid_peaks is None or len(valid_peaks) < 2:
    return data

  # total_intervals = len(valid_intervals)
  # abnormal_intervals_count = np.sum((valid_intervals < 500) | (valid_intervals > 1200))
  # abnormal_intervals_percentage = abnormal_intervals_count / total_intervals if total_intervals > 0 else 0

  # if abnormal_intervals_percentage > 0.15:
  #   segments = []

  #   valid_peaks = valid_peaks[(valid_intervals >= 500) & (valid_intervals <= 1200)]
  #   valid_intervals = valid_intervals[(valid_intervals >= 500) & (valid_intervals <= 1200)]

  #   for i in range(len(valid_peaks) - 1):
  #     start_idx = valid_peaks[i]
  #     end_idx = valid_peaks[i + 1]
  #     segment = data[col].iloc[start_idx: end_idx]
  #     segment.append(segment)

  #   # final_segment = pd.DataFrame(pd.concat(segments))
  #   final_segment = pd.concat(segments, ignore_index=True)

  #   return final_segment

  # else:
  #   return data

  segments = []
  for i in range(len(valid_peaks) - 1):
    start_idx = valid_peaks[i]
    end_idx = valid_peaks[i + 1]

    segment = data.iloc[start_idx: end_idx]
    segments.append(segment)

  final_segment = pd.DataFrame(pd.concat(segments))

  return final_segment

def lowpass_filter(data, cutoff = 5, fs = 64, order = 4):
  filtered_data = pd.DataFrame()
  nyquist = 0.5 * fs
  normal_cutoff = cutoff / nyquist
  b, a = butter(order, normal_cutoff, btype = 'low', analog = False)
  filtered_data['eda'] = filtfilt(b, a, data)
  return filtered_data

def upsample_signal(signal, original_fs, target_fs):
  signal = pd.to_numeric(signal, errors='coerce')
  ratio = target_fs / original_fs
  target_time = np.linspace(0, len(signal) / original_fs, int(len(signal) * ratio))
  original_time = np.linspace(0, len(signal) / original_fs, len(signal))
  spline_interpolator = CubicSpline(original_time, signal)
  upsampled_signal = spline_interpolator(target_time)
  return upsampled_signal

# def upsample_signal(signal, original_fs, target_fs):
#   ratio = target_fs / original_fs
#   upsampled_signal = resample(signal, int(len(signal) * ratio))
#   return upsampled_signal

def gaussian_lowpass_filter(signal, window_size, sigma):
  window = gaussian_filter(window_size, sigma = sigma)
  window /= np.sum(window)
  smoothed_signal = convolve(signal, window, mode= 'same')
  return smoothed_signal

def smooth_signal(signal, window_size = 40, sigma = 400):
    x = np.linspace(-window_size // 2, window_size // 2, window_size)
    window = np.exp(-x**2 / (2 * sigma**2))

    window = window.astype(np.float64)

    window /= np.sum(window)

    smoothed_signal = convolve(signal, window, mode='same')

    return smoothed_signal

def pad_signal_to_length(signal_data, target_length):
  if len(signal_data) < target_length:
    signal_data = np.pad(signal_data, (0, target_length - len(signal_data)), mode='constant')

  elif len(signal_data) > target_length:
    signal_data = signal_data[:target_length]

  return signal_data

def remove_constant_segments(signal, col, threshold = 10**(-10), fs = 64):
    diff = np.abs(np.diff(signal[col].values))
    diff_zero_indices = np.where(diff < threshold)[0]

    filtered_signal = []
    start_idx = None
    min_length = 10000

    for idx in signal.index[:]:
        if idx in diff_zero_indices:
            if start_idx is None:
                start_idx = idx
        else:
            if start_idx is not None:
                if idx - start_idx < min_length:
                    filtered_signal.extend(signal.loc[start_idx:idx, col].values.tolist())
                start_idx = None

            filtered_signal.append(signal.loc[idx, col])

    if start_idx is not None and len(signal) - start_idx < min_length:
        filtered_signal.extend(signal.loc[start_idx:, col].values.tolist())

    filtered_df = pd.DataFrame(filtered_signal, columns=[col])

    return filtered_df

def load_data(data_dir):
  subject_data = {}

  segments = [
    ('Rest 1', 3),
    ('Task 1', 10),
    ('Rest 2', 2),
    ('Task 2', 5),
    ('Rest 3', 2),
    ('Task 3', 3),
    ('Rest 4', 2),
    ('Task 4', None),
    ('Rest 5', 2),
    ('Task 5', 1),
    ('Rest 6', 2),
  ]

  eda_sampling_rate = 4
  bvp_sampling_rate = 64
  hr_sampling_rate = 1

  subjects = [f'subject_{i:02d}' for i in range(1, 30)]
  for subject in subjects:
    eda_path = os.path.join(data_dir, subject, 'EDA.csv')
    bvp_path = os.path.join(data_dir, subject, 'BVP.csv')
    # hr_path = os.path.join(data_dir, subject, 'HR.csv')

    if os.path.exists(eda_path) and os.path.exists(bvp_path):
      eda_data = pd.read_csv(eda_path, names = ['eda'])
      bvp_data = pd.read_csv(bvp_path, names = ['bvp'])
      # hr_data = pd.read_csv(hr_path, names = ['hr'])

      eda_data = interpolate_missing_values(eda_data, col = 'eda', fs = eda_sampling_rate)
      bvp_data = interpolate_missing_values(bvp_data, col = 'bvp', fs = bvp_sampling_rate)
      # hr_data = interpolate_missing_values(hr_data, col = 'hr', fs = hr_sampling_rate)

      # eda_data = split_signal(eda_data['eda'], fs = eda_sampling_rate, interval = 60)
      # bvp_data = split_signal(bvp_data['bvp'], fs = bvp_sampling_rate, interval = 60)

      eda_data = segment_process(eda_data, 'eda', segments, eda_sampling_rate)
      bvp_data = segment_process(bvp_data, 'bvp', segments, bvp_sampling_rate)
      # hr_data = segment_process(hr_data, 'hr', segments, hr_sampling_rate)

      # HR
      # upsampled_hr = upsample_signal(hr_data['hr'], 1, 64)
      # padded_hr = pad_signal_to_length(upsampled_hr, bvp_data.shape[0])

      # EDA
      upsampled_eda = upsample_signal(eda_data['eda'], 4, 64)
      padded_eda = pad_signal_to_length(upsampled_eda, bvp_data.shape[0])
      filtered_eda = lowpass_filter(upsampled_eda, fs = 64)
      # peaks = peak_detection(filtered_eda, 'eda', min_distance = 0.5, min_height = 0.5, fs = 64)
      # valid_peaks, valid_intervals = calculate_peak_to_peak_intervals(filtered_eda, peaks)
      # final_segment_eda = adjust_intervals(filtered_eda, valid_peaks, valid_intervals)
      final_segment_eda = remove_constant_segments(filtered_eda, col = 'eda')
      print(filtered_eda.shape, final_segment_eda.shape)

      # plt.figure(figsize = (15, 7))
      # plt.plot(filtered_eda.index, upsampled_eda, label = 'signal')
      # plt.legend()
      # plt.xlabel('Time')
      # plt.show()

      # plt.figure(figsize = (15, 7))
      # plt.plot(final_segment_eda.index, final_segment_eda['eda'], label='Remove Constant', color='g')
      # plt.legend()
      # plt.xlabel('Time')
      # plt.show()

      # smoothed_eda = smooth_signal(upsampled_eda, window_size = 40, sigma = 400)
      # smoothed_eda = pad_signal_to_length(smoothed_eda, bvp_data.shape[0])

      # plt.figure(figsize = (15, 7))
      # plt.plot(range(len(padded_hr)), padded_hr)
      # plt.show()

      # plt.figure(figsize = (15, 7))
      # plt.plot(range(len(smoothed_eda)), smoothed_eda)
      # plt.show()

      # plt.figure(figsize = (15, 7))
      # plt.plot(bvp_data.index, bvp_data['bvp'])
      # plt.show()

      # BVP
      bvp_data = chevyfilter_process(bvp_data, 'bvp')

      # Merge
      # bvp_data['eda'] = smoothed_eda
      # bvp_data['hr'] = padded_hr

      # Peak Detection
      peaks = peak_detection(bvp_data, 'bvp')
      valid_peaks, valid_intervals = calculate_peak_to_peak_intervals(bvp_data, peaks)
      final_segment_bvp = adjust_intervals(bvp_data, valid_peaks, valid_intervals)
      print(bvp_data.shape, final_segment_bvp.shape)

      # plt.figure(figsize = (15, 7))
      # plt.plot(bvp_data.index, bvp_data['bvp'], label = 'signal')
      # plt.scatter(bvp_data.loc[peaks].index, bvp_data.loc[peaks, 'bvp'], label = 'Detected peaks', color = 'r')
      # plt.plot(final_segment_bvp.index, final_segment_bvp['bvp'], label='Valid Peaks', color='g')
      # plt.legend()
      # plt.title('Signal with Detected Peaks')
      # plt.xlabel('Time')
      # plt.show()

      print(final_segment_eda.index[0], final_segment_eda.index[-1])
      print(final_segment_bvp.index[0], final_segment_bvp.index[-1])

      common_indices = final_segment_eda.index.intersection(final_segment_bvp.index)
      print(common_indices)
      print(min(common_indices), max(common_indices))

      if final_segment_eda.index[0] < final_segment_bvp.index[0]:
        final_segment_eda = final_segment_eda.loc[common_indices.min():]

      elif final_segment_eda.index[0] > final_segment_bvp.index[0]:
        final_segment_bvp = final_segment_bvp.loc[common_indices.min():]

      if final_segment_eda.index[-1] < final_segment_bvp.index[-1]:
        final_segment_bvp = final_segment_bvp.loc[:common_indices.max()]

      elif final_segment_eda.index[-1] > final_segment_bvp.index[-1]:
        final_segment_eda = final_segment_eda.loc[:common_indices.max()]

      print('final_segment_eda', final_segment_eda.index[0], final_segment_eda.index[-1])
      print('final_segment_bvp', final_segment_bvp.index[0], final_segment_bvp.index[-1])

      final_segment = pd.concat([final_segment_eda, final_segment_bvp], axis = 1)
      print('final segment', final_segment.shape)

      # plt.plot(final_segment.index, final_segment['eda'], label = 'final eda', color = 'b')
      # plt.legend()
      # plt.title('Final EDA Signal')
      # plt.xlabel('Time')
      # plt.show()
      # plt.plot(final_segment.index, final_segment['bvp'], label = 'final bvp', color = 'b')
      # plt.legend()
      # plt.title('Final BVP signal')
      # plt.xlabel('Time')
      # plt.show()

      final_segment = final_segment.reset_index()

      subject_data[subject] = final_segment # (bvp_data, smoothed_eda)

  return subject_data

subject_data = load_data(os.getenv('STRESS_RAW_DIR', 'data/Subjects'))

subjects_dir = os.getenv('STRESS_DATA_DIR', 'data/Subjects_processing')

for subject in list(subject_data.keys()):
  
  subjects_path = subjects_dir + '/' + subject
  
  if os.path.exists(subjects_path):
    print('exists..')
  else:
    os.mkdir(subjects_path)
  
  subject_data[subject].to_csv(subjects_path + '/' + 'Subject_data.csv', index = False)