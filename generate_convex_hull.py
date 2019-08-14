# !/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


def find_lowest(points):
    points = np.array(points)
    idx = np.argmax(points[:, 0])
    return points[idx].tolist()


def calculate_ctan(base_point, end_point):
    delta_x = end_point[1] - base_point[1]
    delta_y = end_point[0] - base_point[0]
    return delta_x/(delta_y + 1e-16)


def sort_points(arr, points):
    if len(arr) < 2:
        return points
    else:
        pivot = arr[0]
        base_point = points[0]
        less = [i for i in arr[1:] if i <= pivot]
        left_points = [points[arr.index(i)] for i in less]
        greater = [i for i in arr[1:] if i > pivot]
        right_points = [points[arr.index(i)] for i in greater]
        return sort_points(less, left_points) + [base_point] + sort_points(greater, right_points)


def is_anticlockwise(sta_point, mid_point, end_point):
    front_vector = [mid_point[0] - sta_point[0], mid_point[1] - sta_point[1]]
    after_vector = [end_point[0] - mid_point[0], end_point[1] - mid_point[1]]
    mul = front_vector[0]*after_vector[1] - after_vector[0]*front_vector[1]
    return mul > 0


def calculate_convex_hull(points):
    lowest_point = find_lowest(points)
    points.remove(lowest_point)
    ctan_ls = []
    for point in points:
        if point == lowest_point:
            continue
        ctan_ls.append(calculate_ctan(lowest_point, point))
    points_sorted = sort_points(ctan_ls, points)

    res = [lowest_point, points_sorted[0]]
    for i in range(1, len(points_sorted)):
        point = points_sorted[i]
        while is_anticlockwise(res[-2], res[-1], point) is False:
            res.pop(-1)
            if len(res) <= 2:
                break
        res.append(point)
    res.append(lowest_point)
    return res


if __name__ == '__main__':
    arr = np.array([[0, 0, 0, 1, 1, 1, 0, 0, 0],
                    [0, 0, 1, 1, 1, 1, 1, 1, 0],
                    [0, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 1, 1, 1, 1, 1, 0, 0],
                    [0, 1, 1, 1, 1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 1, 1, 1, 1, 1, 1, 1, 0],
                    [0, 0, 1, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0, 0]])

