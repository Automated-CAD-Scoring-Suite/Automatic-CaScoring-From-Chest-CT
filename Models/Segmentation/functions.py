##
# Functions implemented for loss and optimization
##
import tensorflow as tf
import tensorflow.keras.backend as K


def Dice(y_true, y_pred, smooth=1):
    """
        Dice Coefficient Implementation
        Dice = (2*|y_true * y_pred|)/(|y_true|+|y_pred|)
    :param y_true: True Output
    :param y_pred: Predicted Output
    :param smooth: Smoothing Coefficient
    :return:
    """
    intersection = 2 * K.abs(y_true * y_true)
    dice = (intersection + smooth) / (K.abs(y_true) + K.abs(y_pred) + smooth)
    return np.mean(dice)


def Dice_Loss(y_true, y_pred, smooth=1):
    """
        Dice Loss Implementation
        Dice_loss = 1 - (2*|y_true * y_pred|)/(|y_true|+|y_pred|)
    :param y_true: True Output
    :param y_pred: Predicted Output
    :param smooth: Smoothing Coefficient
    :return: Average Dice Loss over all Classes
    """
    return K.mean(1 - Dice(y_true, y_pred, smooth))


if __name__ == '__main__':
    import numpy as np
    x = np.array([
        [0, 0, 1, 1, 1, 1],
        [0, 0, 1, 0, 0, 1],
        [0, 0, 1, 0, 0, 1],
        [0, 0, 1, 0, 0, 1],
        [0, 0, 1, 0, 0, 1],
        [0, 0, 1, 1, 1, 1],
    ])

    y = np.copy(x)

    t = 2 * np.abs(y * x)
    t = (t + 1) / (np.abs(x) + np.abs(y) + 1)
    print(np.mean(t))
    print(Dice(y, x))
