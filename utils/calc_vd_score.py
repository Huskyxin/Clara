import numpy as np
from sklearn.metrics import roc_curve
import argparse
import json

def calculate_vul_det_score(predictions, ground_truth, target_fpr=0.005):
    """
    Calculate the vulnerability detection score (VD-S) given a tolerable FPR.
    
    Args:
    - predictions: List of model prediction probabilities for the positive class.
    - ground_truth: List of ground truth labels, where 1 means vulnerable class, and 0 means benign class.
    - target_fpr: The tolerable false positive rate.
    
    Returns:
    - vds: Calculated vulnerability detection score given the acceptable .
    - threshold: The classification threashold for vulnerable prediction.
    """
    
    # Calculate FPR, TPR, and thresholds using ROC curve
    fpr, tpr, thresholds = roc_curve(ground_truth, predictions)
    
    # Filter thresholds where FPR is less than or equal to the target FPR
    valid_indices = np.where(fpr <= target_fpr)[0]
    
    # Choose the threshold with the largest FPR that is still below the target FPR, if possible
    if len(valid_indices) > 0:
        idx = valid_indices[-1]  # Last index where FPR is below or equal to target FPR
    else:
        # If no such threshold exists (unlikely), default to the closest to the target FPR
        idx = np.abs(fpr - target_fpr).argmin()
        
    chosen_threshold = thresholds[idx]
    
    # Classify predictions based on the chosen threshold
    classified_preds = [1 if pred >= chosen_threshold else 0 for pred in predictions]
    
    # Calculate VD-S
    fn = sum([1 for i in range(len(ground_truth)) if ground_truth[i] == 1 and classified_preds[i] == 0])
    tp = sum([1 for i in range(len(ground_truth)) if ground_truth[i] == 1 and classified_preds[i] == 1])
    vds = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    return vds, chosen_threshold

if __name__ == "__main__":

    import pandas as pd

    # 读取 CSV 文件
    csv_file_path = 'save/Best_Pred_idxs/DiverseVul/codebert+pool-default.csv'  # 替换为你的 CSV 文件路径
    df = pd.read_csv(csv_file_path)

    # 提取名为 True_Label 和 Pred_scores 的列
    ground_truth = df['True_Label']
    pred = df['Pred_scores']

        
    target_fpr = 1
    vds, threshold = calculate_vul_det_score(pred, ground_truth, target_fpr=target_fpr)
    print(f"VD-S: {vds}, Picked FPR: {target_fpr}, Threshold: {threshold}")
