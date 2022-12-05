# Import Required Packages
import numpy as np


def find_roi_2D(s):
    """

    :param s:
    :return:
    """

    # rotate -90
    s_rotated = np.rot90(s, k=3)

    # flip slice
    s_fliped = np.flip(s, axis=0)
    s_rotated_fliped = np.flip(s_rotated, axis=0)

    # Get up and down coordinates
    y1 = np.unravel_index(np.argmax(s, axis=None), s.shape)
    y2 = np.unravel_index(np.argmax(s_fliped, axis=None), s.shape)

    x1 = np.unravel_index(np.argmax(s_rotated, axis=None), s.shape)
    x2 = np.unravel_index(np.argmax(s_rotated_fliped, axis=None), s.shape)

    # return x1, x2, y1, y2 of image
    return x1[0], s.shape[1] - x2[0], y1[0], s.shape[0] - y2[0]


def get_coords(Slices: list):
    """
    Returns shape[1] then shape[0]
    :param Slices: list of 3 slices
    :return: list [x1,x2,y1,y2]
    """

    # Initialize coordinates
    x1 = list()
    x2 = list()
    y1 = list()
    y2 = list()

    # Find ROI in each slice
    for _slice in Slices:
        pnt1, pnt2, pnt3, pnt4 = find_roi_2D(_slice)
        x1.append(pnt1)
        x2.append(pnt2)
        y1.append(pnt3)
        y2.append(pnt4)

    # Return shape[1] then shape[0]
    return [min(x1), max(x2), min(y1), max(y2)]


def GetCoords(Segmentation, Partial=True):
    """
    Get Cropping Directions In The 3 Planes For The Given Segmentation of Shape (Z, X, Y)
    """
    if Partial:
        # Get Coordinates For Each View
        AxCoor = [int(i) for i in
                  get_coords([Segmentation[0][0][0, :, :], Segmentation[0][0][1, :, :], Segmentation[0][0][2, :, :]])]
        SagCoor = [int(i) for i in
                   get_coords([Segmentation[1][0][0, :, :], Segmentation[1][0][1, :, :], Segmentation[1][0][2, :, :]])]
        CorCoor = [int(i) for i in
                   get_coords([Segmentation[2][0][0, :, :], Segmentation[2][0][1, :, :], Segmentation[2][0][2, :, :]])]
        CoordinatesList = [AxCoor, SagCoor, CorCoor]

        # Determine Correct Cropping Coordinates For Each Dimension
        # Coordinates = [[Xmin, Xmax, Ymin, Ymax],[Zmin,Zmax, Ymin, Ymax],[Zmin, Zmax, Xmin,Xmax]]

        x1 = np.minimum(CoordinatesList[0][0], CoordinatesList[2][0])
        x2 = np.maximum(CoordinatesList[0][1], CoordinatesList[2][1])
        y1 = np.minimum(CoordinatesList[0][2], CoordinatesList[1][0])
        y2 = np.maximum(CoordinatesList[0][3], CoordinatesList[1][1])
        z1 = np.minimum(CoordinatesList[1][2], CoordinatesList[2][2])
        z2 = np.maximum(CoordinatesList[1][3], CoordinatesList[2][3])
        Coordinates32 = [z1, z2, x1, x2, y1, y2]

    else:
        Location = np.where(Segmentation > 0)
        Coordinates32 = [np.min(Location[0]), np.max(Location[0]),
                         np.min(Location[1]), np.max(Location[1]),
                         np.min(Location[2]), np.max(Location[2])]

    # x1 = np.minimum(CoordinatesList[0][0], CoordinatesList[2][0])
    # x2 = np.maximum(CoordinatesList[0][1], CoordinatesList[2][1])
    # y1 = np.minimum(CoordinatesList[0][2], CoordinatesList[1][0])
    # y2 = np.maximum(CoordinatesList[0][3], CoordinatesList[1][1])
    # z1 = np.minimum(CoordinatesList[1][2], CoordinatesList[2][2])
    # z2 = np.maximum(CoordinatesList[1][3], CoordinatesList[2][3])

    Coordinates = [int(i) for i in Coordinates32]

    return Coordinates
