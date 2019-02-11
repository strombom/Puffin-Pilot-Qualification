
import math
import numpy as np
from enum import IntEnum
from numba import jit, njit, jitclass
from numba import int64, float64

from clusters import MAX_POINTS
from common import norm
from common import make_line_from_points
from common import get_farthest_point_from_point
from common import get_farthest_point_from_line
from common import get_points_close_to_line


def match_corners(corners):
    return jit_match_corners(corners)



#@njit
def jit_match_corners(corners):


    for corner in corners:
        print("---", corner)

        #for corner in cluster:
        #    print(point)

        quit()

    return []
