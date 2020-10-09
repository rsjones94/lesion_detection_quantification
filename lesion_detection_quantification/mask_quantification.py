import numpy as np
import scipy

from sklearn import metrics
from scipy import spatial
from sklearn.neighbors import DistanceMetric


def dice_coef(y_true, y_pred):
    """
    Courtesy Zabir Al Nazi on StackOverflow
    """
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    smooth = 0.0001
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def volumetric_similarity(y_true, y_pred):
    
    true_vol = y_true.sum()
    pred_vol = y_pred.sum()
    
    diff = abs(true_vol - pred_vol)
    
    return diff / (true_vol + pred_vol)
    

def quantify_mask_quality(y_true, y_pred):
    """
    Returns a number of metrics for mask quality as a dictionary
    
    Metrics chosen were inspired by Taha and Hanbury, Metrics for evaluating 3D medical image segmentation: analysis, selection, and tool
    
    Parameters
    ----------
    y_true : TYPE
        DESCRIPTION.
    y_pred : TYPE
        DESCRIPTION.
    
    Returns
    -------
    dict of metrics.
    
    """
    
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    
    f1 = metrics.f1_score(y_true_f, y_pred_f)
    jac = metrics.jaccard_score(y_true_f, y_pred_f)
    
    tn, fp, fn, tp = (metrics.confusion_matrix(y_true_f, y_pred_f).ravel())
    tpr = tp / (tp+fn)
    tnr = tn / (tn+fp)
    fpr = fp / (fp+tn)
    fnr = fn / (fn+tp)
    
    vol_sim = volumetric_similarity(y_true_f, y_pred_f)
    adj_rand = metrics.adjusted_rand_score(y_true_f, y_pred_f)
    adj_mut_info = metrics.adjusted_mutual_info_score(y_true_f, y_pred_f)
    kappa = metrics.cohen_kappa_score(y_true_f, y_pred_f)
    #roc = metrics.roc_auc_score(y_true, y_pred)
    
    the_dict = {'f1_score': f1,
                'jaccard_index': jac,
                'true_negative_rate': tnr,
                'true_positive_rate': tpr,
                'false_negative_rate': fnr,
                'false_positive_rate': fpr,
                'volumetric_similarity': vol_sim,
                'adjusted_rand_index': adj_rand,
                'adjusted_mutual_information': adj_mut_info,
                'cohens_kappa': kappa}
    
    # 'area_under_roc_curve': roc
    
    return the_dict