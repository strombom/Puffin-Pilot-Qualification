
import math
import numpy as np
from numba import jit, njit, jitclass
from numba import int64, float64, boolean

max_cluster_count = 10
MAX_POINTS = 15
max_point_dist = 34.0


def make_clusters(points):
    clusters = jit_make_clusters(points)

    #print("make_clusters", clusters)

    # Return the clusters
    return clusters # [cluster.points for cluster in clusters]

#@njit
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
            if try_to_append(cluster, points[pidx]):
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
                if try_to_merge(clusters[idx_i], clusters[idx_j]):
                    del clusters[idx_j]
                    merging = True
                    break
            if merging:
                break

    split_clusters(clusters)

    #for cluster in clusters:
    #    cluster.shrink()

    return clusters


#@njit
def split_clusters(clusters):
    if len(clusters) != 2:
        return

    for i in range(2):
        if clusters[i].points_count < 7:
            return

    points_left = np.empty((MAX_POINTS, 2))
    points_right = np.empty((MAX_POINTS, 2))
    cog = np.empty((2))

    for i in range(2):
        points_left_count, points_right_count = 0, 0

        cog[:] = np.sum(clusters[i].points[0:clusters[i].points_count], axis=0)
        cog /= clusters[i].points_count

        split_x = cog[0]
        for idx in range(clusters[i].points_count):
            point = clusters[i].points[idx]
            if abs(point[0] - split_x) < 8:
                continue
            if point[0] < split_x:
                points_left[points_left_count] = point
                points_left_count += 1
            else:
                points_right[points_right_count] = point
                points_right_count += 1

        clusters[i].points[:] = points_left
        clusters[i].points_count = points_left_count
        
        new_cluster = Cluster(points_right[0])
        new_cluster.points[:] = points_right
        new_cluster.points_count = points_right_count
        clusters.append(new_cluster)


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

#@jitclass(cluster_spec)
class Cluster(object):
    # A cluster is a collection of centroids with close proximity,
    #  clusters are later turned into corners.

    def __init__(self, point):
        self.points = np.zeros((MAX_POINTS, 2), dtype=np.float64)
        self.points_count = 1
        self.max_y = point[1]
        self.points[0] = point
        self.active = True

        self.box_xmin = point[0] - max_point_dist / 2
        self.box_xmax = point[0] + max_point_dist / 2
        self.box_ymin = point[1] - max_point_dist / 2
        self.box_ymax = point[1] + max_point_dist / 2


#@njit
def try_to_append(cluster, point):
    if not cluster.active:
        return False

    if point[1] - cluster.max_y > max_point_dist:
        # This cluster is too far away from where we are looking, 
        #  disable it.
        cluster.active = False
        return False

    distance = distance_from_points_to_point(cluster.points, cluster.points_count, point)
    if distance > max_point_dist:
        # Point is too far away from this cluster.
        return False

    if cluster.points_count == MAX_POINTS:
        # We silently ignore if too many points are appended
        #  to a cluster. The cluster is malformed and we can
        #  only hope that subsequent steps will be able to
        #  fit one or two lines to the cluster points.
        return True

    cluster.points[cluster.points_count] = point
    cluster.points_count += 1
    cluster.max_y = point[1]

    cluster.box_xmin = min(cluster.box_xmin, point[0] - max_point_dist / 2)
    cluster.box_xmax = max(cluster.box_xmax, point[0] + max_point_dist / 2)
    cluster.box_ymin = min(cluster.box_ymin, point[1] - max_point_dist / 2)
    cluster.box_ymax = max(cluster.box_ymax, point[1] + max_point_dist / 2)
    return True

#@njit
def try_to_merge(cluster1, cluster2):
    if cluster1.box_xmin > cluster2.box_xmax or cluster2.box_xmin > cluster1.box_xmax or \
       cluster1.box_ymin > cluster2.box_ymax or cluster2.box_ymin > cluster1.box_ymax:
        # No overlap, don't try to merge
        return False
    
    merge = False
    for pidx in range(cluster2.points_count):
        if distance_from_points_to_point(cluster1.points, cluster1.points_count, cluster2.points[pidx]) <= max_point_dist:
            merge = True
            break
    if merge:
        for idx in range(cluster2.points_count):
            if cluster1.points_count == MAX_POINTS:
                break
            cluster1.points[cluster1.points_count] = cluster2.points[idx]
            cluster1.points_count += 1
        cluster1.box_xmin = min(cluster1.box_xmin, cluster2.box_xmin)
        cluster1.box_xmax = max(cluster1.box_xmax, cluster2.box_xmax)
        cluster1.box_ymin = min(cluster1.box_ymin, cluster2.box_ymin)
        cluster1.box_ymax = max(cluster1.box_ymax, cluster2.box_ymax)
        return True
    else:
        return False

@njit
def distance_from_points_to_point(points, points_count, point):
    if points_count == 0:
        return float(0.0)

    min_distance = 1.0e9
    for pidx in range(points_count):
        v = point - points[pidx]
        v_norm = math.sqrt(v[0]*v[0] + v[1]*v[1])
        min_distance = min(min_distance, v_norm)

    return float(min_distance)
