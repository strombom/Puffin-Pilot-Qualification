
import math
import numpy as np
from numba import jit, njit, jitclass
from numba import int64, float64, boolean

max_cluster_count = 10
MAX_POINTS = 15
max_point_dist_1 = 34.0
max_point_dist_2 = 43.0


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
    merge_singles(clusters)
    fix_outliers(clusters)
    fix_malformed(clusters)

    #for cluster in clusters:
    #    cluster.shrink()

    return clusters

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
    # A cluster is a collection of centroids with close proximity,
    #  clusters are later turned into corners.

    def __init__(self, point):
        self.points = np.zeros((MAX_POINTS, 2), dtype=np.float64)
        self.points_count = 1
        self.max_y = point[1]
        self.points[0] = point
        self.active = True

        self.box_xmin = point[0] - max_point_dist_1 / 2
        self.box_xmax = point[0] + max_point_dist_1 / 2
        self.box_ymin = point[1] - max_point_dist_1 / 2
        self.box_ymax = point[1] + max_point_dist_1 / 2


#@njit
def split_clusters(clusters):
    big_cluster_count = 0
    for cluster in clusters:
        if cluster.points_count > 4:
            big_cluster_count += 1

    if big_cluster_count < 4:
        points_left = np.empty((MAX_POINTS, 2))
        points_right = np.empty((MAX_POINTS, 2))

        for cluster in clusters:
            make_boundary_box(cluster)

        splitting = True
        while splitting:
            splitting = False

            for ci, cluster in enumerate(clusters):

                if cluster.points_count < 9:
                    continue

                xmin = cluster.box_xmin + max_point_dist_1 / 2
                xmax = cluster.box_xmax - max_point_dist_1 / 2
                ymin = cluster.box_ymin + max_point_dist_1 / 2
                ymax = cluster.box_ymax - max_point_dist_1 / 2
                if xmax < xmin or ymax < ymin:
                    continue

                maxwidth = 77
                if cluster.points_count > 11:
                    maxwidth = 95

                if xmax - xmin > 40 and xmax - xmin < maxwidth and \
                   ymax - ymin > 30 and ymax - ymin < 65:
                    split_x = (xmax + xmin) / 2
                    print("split_x", split_x)
                    x_threshold = (xmax - xmin) / 7.0

                    points_left_count, points_right_count = 0, 0
                    for idx in range(cluster.points_count):
                        point = cluster.points[idx]
                        if abs(point[0] - split_x) < x_threshold:
                            continue
                        if point[0] < split_x:
                            points_left[points_left_count] = point
                            points_left_count += 1
                        else:
                            points_right[points_right_count] = point
                            points_right_count += 1

                    print("pointcount", points_left_count, points_right_count)

                    if points_left_count > 2:
                        cluster.points[:] = points_left
                        cluster.points_count = points_left_count
                        make_boundary_box(cluster)

                        if points_right_count > 2:
                            new_cluster = Cluster(points_right[0])
                            new_cluster.points[:] = points_right
                            new_cluster.points_count = points_right_count
                            make_boundary_box(new_cluster)
                            clusters.append(new_cluster)
                            splitting = True

                    elif points_right_count > 2:
                        cluster.points[:] = points_right
                        cluster.points_count = points_right_count
                        make_boundary_box(cluster)


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


def merge_singles(clusters):
    avg_point_distance = 0
    point_count = 0

    for cluster in clusters:
        if cluster.points_count < 2:
            continue




    for cluster in clusters:
        if cluster.points_count != 1:
            continue


        print(cluster.points[0])



@njit
def fix_outliers(clusters):
    if len(clusters) < 2:
        return

    big_cluster_count = 0
    largest_cluster_count = 0
    largest_cluster_idx = 0
    for idx, cluster in enumerate(clusters):
        if cluster.points_count > 2:
            big_cluster_count += 1
        if cluster.points_count > largest_cluster_count:
            largest_cluster_count = cluster.points_count
            largest_cluster_idx = idx

    if big_cluster_count > 4:
        cluster_sizes = []
        for cluster in clusters:
            cluster_sizes.append(cluster.points_count)
        cluster_sizes.sort()
        threshold = cluster_sizes[-4]
        new_clusters = []
        for cluster in clusters:
            if cluster.points_count >= threshold:
                new_clusters.append(cluster)
        clusters[:] = new_clusters

    if big_cluster_count > 2:
        return

    closest_cluster_idx = 0
    closest_cluster_distance = 1e9
    largest_cluster_center = np.sum(clusters[largest_cluster_idx].points[0:clusters[largest_cluster_idx].points_count], axis = 0) / clusters[largest_cluster_idx].points_count

    for idx, cluster in enumerate(clusters):
        if idx == largest_cluster_idx:
            continue

        if big_cluster_count > 1 and cluster.points_count < 3:
            continue

        point1 = np.sum(cluster.points[0:cluster.points_count], axis = 0) / cluster.points_count
        v = point1 - largest_cluster_center
        distance = math.sqrt(v[0]*v[0] + v[1]*v[1])
        if distance < closest_cluster_distance:
            closest_cluster_distance = distance
            closest_cluster_idx = idx

    cluster_1 = clusters[largest_cluster_idx]
    cluster_2 = clusters[closest_cluster_idx]

    xmin, xmax = 1e9, 0
    ymin, ymax = 1e9, 0

    for cluster in [cluster_1, cluster_2]:
        pcog = np.sum(cluster.points[0:cluster.points_count], axis = 0)
        pcog = pcog / cluster.points_count
        xmin = min(xmin, pcog[0])
        xmax = max(xmax, pcog[0])
        ymin = min(ymin, pcog[1])
        ymax = max(ymax, pcog[1])

    width, height = xmax - xmin, ymax - ymin
    if height < width:
        ratio = 1.0
        if width < 150:
            ratio = 2.0
        if ymax < 430:
            ymax = ymin + width * ratio
        else:
            ymin = ymax - width * ratio
    else:
        # Check if there is a third point
        if width < 50:
            xmax += 90

    ymax += 20
    ymax = min(ymax, 850)
    ymin = max(ymin, 40)

    if width < 50:
        xmax = xmax + 25
        xmin = xmin - 25

    clusters.clear()
    for idx, point in enumerate(((xmin, ymin), (xmax, ymin), (xmin, ymax), (xmax, ymax))):
        dy = -((idx // 2) * 2 - 1) * 10
        dx = -((idx %  2) * 2 - 1) * 10
        cluster = Cluster(point)
        cluster.points[1] = (dx + point[0], point[1])
        cluster.points[2] = (2 * dx + point[0], point[1])
        cluster.points[3] = (point[0], dy + point[1])
        cluster.points[4] = (point[0], 2 * dy + point[1])
        cluster.points_count = 5
        clusters.append(cluster)


@njit
def fix_malformed(clusters):
    cog = np.zeros(2, np.float64)
    cog_count = 0

    for cluster in clusters:
        for idx in range(cluster.points_count):
            cog += cluster.points[idx]
            cog_count += 1

    cog /= cog_count

    for cluster in clusters:
        if cluster.points_count < 6:
            continue

        xmin, xmax = 1e9, 0
        ymin, ymax = 1e9, 0
        for i in range(cluster.points_count):
            xmin = min(xmin, cluster.points[i][0])
            xmax = max(xmax, cluster.points[i][0])
            ymin = min(ymin, cluster.points[i][1])
            ymax = max(ymax, cluster.points[i][1])

        if ymin < cog[1]:
            continue

        width, height = xmax - xmin, ymax - ymin
        if width > 12 or height < 46 or height / width < 5:
            continue

        new_ymax = ymin + height * 0.75

        new_points = np.zeros((MAX_POINTS, 2), dtype=np.float64)
        new_points_count = 0

        for idx in range(cluster.points_count):
            if cluster.points[idx][1] < new_ymax:
                new_points[new_points_count] = cluster.points[idx]
                new_points_count += 1

        cluster.points[:] = new_points
        cluster.points_count = new_points_count



@njit
def make_boundary_box(cluster):
    xmin, xmax = 1e9, 0
    ymin, ymax = 1e9, 0
    for i in range(cluster.points_count):
        xmin = min(xmin, cluster.points[i][0])
        xmax = max(xmax, cluster.points[i][0])
        ymin = min(ymin, cluster.points[i][1])
        ymax = max(ymax, cluster.points[i][1])
    cluster.box_xmin = xmin - max_point_dist_1 / 2
    cluster.box_xmax = xmax + max_point_dist_1 / 2
    cluster.box_ymin = ymin - max_point_dist_1 / 2
    cluster.box_ymax = ymax + max_point_dist_1 / 2

@njit
def try_to_append(cluster, point):
    if not cluster.active:
        return False

    if point[1] - cluster.max_y > max_point_dist_1:
        # This cluster is too far away from where we are looking, 
        #  disable it.
        cluster.active = False
        return False

    distance = distance_from_points_to_point(cluster.points, cluster.points_count, point)
    if distance > max_point_dist_1:
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

    cluster.box_xmin = min(cluster.box_xmin, point[0] - max_point_dist_1 / 2)
    cluster.box_xmax = max(cluster.box_xmax, point[0] + max_point_dist_1 / 2)
    cluster.box_ymin = min(cluster.box_ymin, point[1] - max_point_dist_1 / 2)
    cluster.box_ymax = max(cluster.box_ymax, point[1] + max_point_dist_1 / 2)
    return True

@njit
def try_to_merge(cluster1, cluster2):
    if cluster1.box_xmin > cluster2.box_xmax or cluster2.box_xmin > cluster1.box_xmax or \
       cluster1.box_ymin > cluster2.box_ymax or cluster2.box_ymin > cluster1.box_ymax:
        # No overlap, don't try to merge.
        return False
    
    for pidx in range(cluster2.points_count):
        if distance_from_points_to_point(cluster1.points, cluster1.points_count, cluster2.points[pidx]) <= max_point_dist_1:
            # Overlap, merge.
            for idx in range(cluster2.points_count):
                if cluster1.points_count == MAX_POINTS:
                    break
                cluster1.points[cluster1.points_count] = cluster2.points[idx]
                cluster1.points_count += 1
            make_boundary_box(cluster1)
            return True

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
