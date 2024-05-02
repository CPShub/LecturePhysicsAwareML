"""
Lecture "Physics-aware Machine Learning" (SoSe 23/24)
Task 1: Feed-Forward Neural Networks

==================

Authors: Jasper O. Schommartz
         
04/2024
"""
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

def evaluate(Qf, Qf_pred, QKQ, QKQ_pred):
    ''' Performs the model evaluation '''

    # Evaluate prediction
    n = Qf.shape[1]
    err = np.zeros([2, n])
    for i in range(n):
        err[0, i] = mean_squared_error(Qf[:, i], Qf_pred[:, i])
        err[1, i] = r2_score(Qf[:, i], Qf_pred[:, i])
    cols = [f'Qf{i+1}' for i in range(n)]
    rows = ['MSE', 'R2']
    df_y = pd.DataFrame(data=err, columns=cols, index=rows)

    nn = QKQ.shape[1]
    err = np.zeros([2, nn])
    for i in range(nn):
        err[0, i] = mean_squared_error(QKQ[:, i], QKQ_pred[:, i])
        err[1, i] = r2_score(QKQ[:, i], QKQ_pred[:, i])
    cols = [f'QKQ{i+1}' for i in range(nn)]
    rows = ['MSE', 'R2']
    df_dy = pd.DataFrame(data=err, columns=cols, index=rows)

    return df_y, df_dy