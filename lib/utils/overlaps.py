# -*- coding: utf-8 -*-
# @Time    : 2017/10/25 0025 11:10
# @Author  : xiangyaojun
# @Email   : maisca1920@163.com
# @File    : overlaps.py

import numpy as np

def bbox_overlaps(boxes,query_boxes):
    """
        Parameters
        ----------
        boxes: (N, 4) ndarray of float
        query_boxes: (K, 4) ndarray of float
        Returns
        -------
        overlaps: (N, K) ndarray of overlap between boxes and query_boxes
        """
    N = len(boxes)
    K = len(query_boxes)
    overlaps = np.zeros((N, K))
    for k in range(K):
        for n in range(N):
            overlaps[n, k] = IOU(boxes[n], query_boxes[k])

    return overlaps

def IOU(A, B):
    """
    计算两个boxes的IOU值
    Args:
        矩形A = [xmin, ymin, xmax, ymax]
        矩形B = [xmin, ymin, xmax, ymax]
    Returns:
        矩形A和矩形B的IOU值
    """
    W = min(A[3], B[3]) - max(A[1], B[1])
    H = min(A[2], B[2]) - max(A[0], B[0])
    if W <= 0 or H <= 0: #不存在交集
        return 0
    A_area = (A[3] - A[1]) * (A[2] - A[0])  #矩形A的面积
    B_area = (B[3] - B[1]) * (B[2] - B[0])  #矩形B的面积
    cross_area = W * H  #交集的面积
    return cross_area / (A_area + B_area - cross_area)