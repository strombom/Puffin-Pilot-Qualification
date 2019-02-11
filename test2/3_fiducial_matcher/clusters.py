
import math
import numpy as np
from numba import jit, njit, jitclass
from numba import int64, float64, boolean

from common import distance_from_point_to_points


max_cluster_count = 10
MAX_POINTS = 15
max_point_dist = 30.0


def make_clusters(points):
    clusters = jit_make_clusters(points)

    # Return the clusters
    return clusters # [cluster.points for cluster in clusters]


cluster_spec = [
    ('points',       float64[:,:]),
    ('points_count', int64),
    ('max_y',        float64),
    ('active',       boolean),
    ('box_xmin',     float64),
    ('box_xmax',     float64),
    ('box_ymin',     float64),
    ('box_ymax',     float64)
]

@jitclass(cluster_spec)
class Cluster(object):
    def __init__(self, point):
        self.points = np.zeros((MAX_POINTS, 2), dtype=np.float64)
        self.points_count = 1
        self.max_y = point[1]
        self.points[0] = point
        self.active = True

        self.box_xmin = point[0] - max_point_dist * 0.5
        self.box_xmax = point[0] + max_point_dist * 0.5
        self.box_ymin = point[1] - max_point_dist * 0.5
        self.box_ymax = point[1] + max_point_dist * 0.5

    def try_to_append(self, point):
        if not self.active:
            return False

        if point[1] - self.max_y > max_point_dist:
            # This cluster is too far away from where we are looking, 
            #  disable it.
            self.active = False
            return False

        distance = distance_from_point_to_points(point, self.points, self.points_count)
        if distance > max_point_dist:
            # Point is too far away from this cluster.
            return False

        if self.points_count == MAX_POINTS:
            # We silently ignore if too many points are appended
            #  to a cluster. The cluster is malformed and we can
            #  only hope that subsequent steps will be able to
            #  fit one or two lines to the cluster points.
            return True

        self.points[self.points_count] = point
        self.points_count += 1
        self.max_y = point[1]

        self.box_xmin = min(self.box_xmin, point[0] - max_point_dist * 0.5)
        self.box_xmax = max(self.box_xmax, point[0] + max_point_dist * 0.5)
        self.box_ymin = min(self.box_ymin, point[1] - max_point_dist * 0.5)
        self.box_ymax = max(self.box_ymax, point[1] + max_point_dist * 0.5)
        return True

    def try_to_merge(self, cluster):
        if not (self.box_xmax >= cluster.box_xmin and cluster.box_xmax >= self.box_xmin and \
                self.box_ymax >= cluster.box_ymin and cluster.box_ymax >= self.box_ymin):
            # No overlap, don't try to merge
            return False
        
        merge = False
        for pidx in range(cluster.points_count):
            if distance_from_point_to_points(cluster.points[pidx], self.points, self.points_count) <= max_point_dist:
                merge = True
                break
        if merge:
            for idx in range(cluster.points_count):
                if self.points_count == MAX_POINTS:
                    break
                self.points[self.points_count] = cluster.points[idx]
                self.points_count += 1
            return True
        else:
            return False

    def get_center_of_gravity(self):
        return np.sum(self.points, axis=0) / self.points_count

    def shrink(self):
        self.points = self.points[0:self.points_count]


@njit
def jit_make_clusters(points):
    # The points are collected form top to bottom, we compare
    #  new points only with newly formed clusters to reduce
    #  the number of distance evaluations.

    clusters = []

    # Form clusters
    for pidx in range(points.shape[0]):
        # Check if point belongs to an existing cluster
        found_a_friendly_cluster = False
        for cluster in clusters:
            if cluster.try_to_append(points[pidx]):
                found_a_friendly_cluster = True
                break

        # No cluster found, build new home
        if not found_a_friendly_cluster:
            new_home = Cluster(points[pidx])
            clusters.append(new_home)

    # Merge clusters
    merging = True
    while merging:
        merging = False
        for idx_i in range(len(clusters)):
            for idx_j in range(idx_i + 1, len(clusters)):
                if clusters[idx_i].try_to_merge(clusters[idx_j]):
                    del clusters[idx_j]
                    merging = True
                    break
            if merging:
                break

    for cluster in clusters:
        cluster.shrink()

    return clusters
