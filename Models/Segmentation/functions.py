import tensorflow.keras.backend as bck


def dice_coef(y_true, y_pred, smooth=1.):
    y_true_f = bck.flatten(y_true)
    y_pred_f = bck.flatten(y_pred)
    intersection = bck.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (bck.sum(y_true_f) + bck.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def tversky_index(y_true, y_pred):
    # generalization of dice coefficient algorithm
    #   alpha corresponds to emphasis on False Positives
    #   beta corresponds to emphasis on False Negatives (our focus)
    #   if alpha = beta = 0.5, then same as dice
    #   if alpha = beta = 1.0, then same as IoU/Jaccard
    alpha = 0.5
    beta = 0.5
    y_true_f = bck.flatten(y_true)
    y_pred_f = bck.flatten(y_pred)
    intersection = bck.sum(y_true_f * y_pred_f)
    return (intersection) / (intersection + alpha * (bck.sum(y_pred_f*(1. - y_true_f))) + beta *  (bck.sum((1-y_pred_f)*y_true_f)))


def tversky_index_loss(y_true, y_pred):
    return -tversky_index(y_true, y_pred)
