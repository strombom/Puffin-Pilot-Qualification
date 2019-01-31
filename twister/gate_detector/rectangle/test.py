
import math
import numpy as np

from main import line_intersection_angle

"""
def line_intersection_angle(l1, l2):
    v1 = l1[1] - l1[0]
    v2 = l2[1] - l2[0]
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    det = v1[0] * v2[1] - v1[1] * v2[0]
    return math.atan2(det, dot)
"""

l1 = np.array([[7, 5], [6, 4]])
l2 = np.array([[6, 5], [6, 4]])
l3 = np.array([[5, 5], [6, 4]])
l4 = np.array([[5, 4], [6, 4]])
l5 = np.array([[5, 3], [6, 4]])
l6 = np.array([[6, 3], [6, 4]])
l7 = np.array([[7, 3], [6, 4]])
l8 = np.array([[10, 3.99], [6, 4]])
l9 = np.array([[10, 4.01], [6, 4]])
l10 = np.array([[9, 4], [10, 4]])


a = line_intersection_angle(l1, l10)
b = line_intersection_angle(l2, l10)
c = line_intersection_angle(l3, l10)
d = line_intersection_angle(l4, l10)
e = line_intersection_angle(l5, l10)
f = line_intersection_angle(l6, l10)
g = line_intersection_angle(l7, l10)
h = line_intersection_angle(l8, l10)
i = line_intersection_angle(l9, l10)

print(a, b, c, d, e, f, g, h, i)
