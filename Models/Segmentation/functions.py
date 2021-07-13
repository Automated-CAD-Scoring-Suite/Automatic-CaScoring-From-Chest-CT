import tensorflow.keras.backend as bck


def dice_coef(y_true, y_pred, smooth=1.):
    y_true_f = bck.flatten(y_true)
    y_pred_f = bck.flatten(y_pred)
    intersection = bck.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (bck.sum(y_true_f) + bck.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)