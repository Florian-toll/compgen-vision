"""
Metrics module for evaluating model performance and ranking samples.

This module provides various metrics for evaluating model performance
and ranking OOD samples.
"""

from typing import Callable

import numpy as np


def calculate_mse(predictions: np.ndarray, ground_truth: np.ndarray) -> np.ndarray:
  """
  Calculate Mean Squared Error for each sample.

  Args:
      predictions: Predicted values
      ground_truth: Ground truth values

  Returns:
      Array of MSE values for each sample
  """
  return np.mean((predictions - ground_truth) ** 2, axis=1)


def calculate_mae(predictions: np.ndarray, ground_truth: np.ndarray) -> np.ndarray:
  """
  Calculate Mean Absolute Error for each sample.

  Args:
      predictions: Predicted values
      ground_truth: Ground truth values

  Returns:
      Array of MAE values for each sample
  """
  return np.mean(np.abs(predictions - ground_truth), axis=1)


def calculate_cosine_distance(
  predictions: np.ndarray, ground_truth: np.ndarray
) -> np.ndarray:
  """
  Calculate cosine distance for each sample.

  Args:
      predictions: Predicted values
      ground_truth: Ground truth values

  Returns:
      Array of cosine distances for each sample
  """
  pred_norm = np.sqrt(np.sum(predictions**2, axis=1, keepdims=True))
  gt_norm = np.sqrt(np.sum(ground_truth**2, axis=1, keepdims=True))

  pred_norm = np.where(pred_norm == 0, 1e-10, pred_norm)
  gt_norm = np.where(gt_norm == 0, 1e-10, gt_norm)

  pred_normalized = predictions / pred_norm
  gt_normalized = ground_truth / gt_norm

  cosine_sim = np.sum(pred_normalized * gt_normalized, axis=1)

  return 1 - cosine_sim


def calculate_binary_accuracy(
  predictions: np.ndarray, ground_truth: np.ndarray, threshold: float = 0.5
) -> np.ndarray:
  """
  Calculate binary accuracy for each sample.

  Args:
      predictions: Predicted values
      ground_truth: Ground truth values (binary)
      threshold: Threshold for binarizing predictions

  Returns:
      Array of binary accuracy values for each sample
  """
  binary_preds = (predictions > threshold).astype(int)
  return np.mean(binary_preds == ground_truth, axis=1)


def calculate_logloss(
  predictions: np.ndarray,
  ground_truth: np.ndarray,
  eps: float = 1e-12,
) -> np.ndarray:
  """
  Mean binary cross-entropy for *each* sample.

  Returned shape: (N,) so it slots straight into `rank_ood_samples`.
  """
  p = np.clip(predictions, eps, 1 - eps)  # numeric safety
  loss = -(ground_truth * np.log(p) + (1 - ground_truth) * np.log(1 - p))
  return np.mean(loss, axis=1)


# Define a mapping of metric names to functions
METRIC_FUNCTIONS = {
  'mse': calculate_mse,
  'mae': calculate_mae,
  'cosine': calculate_cosine_distance,
  'binary_accuracy': calculate_binary_accuracy,
  'logloss': calculate_logloss,
}


def get_metric(metric_name: str) -> Callable:
  """
  Get a metric function by name.

  Args:
      metric_name: Name of the metric

  Returns:
      Metric function

  Raises:
      ValueError: If metric name is not recognized
  """
  metric_name = metric_name.lower()
  if metric_name not in METRIC_FUNCTIONS:
    raise ValueError(
      f'Unknown metric: {metric_name}. Available metrics: {list(METRIC_FUNCTIONS.keys())}'
    )
  return METRIC_FUNCTIONS[metric_name]
