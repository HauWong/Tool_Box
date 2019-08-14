# !/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


def quick_sort(arr):
    if type(arr) is not list:
        ls = np.array(arr).tolist()
    else:
        ls = arr

    if len(ls) < 2:
        return ls
    else:
        pivot = ls[0]
        less = [_ for _ in arr if _ <= pivot]
        greater = [_ for _ in arr if _ > pivot]
        return quick_sort(less) + [pivot] + quick_sort(greater)