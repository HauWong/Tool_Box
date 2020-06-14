# !/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import random


def shuffle_samples(sample_ls):
    res = sample_ls
    return res


def divide_samples(file_dir, save_dir, name='',
                   types=('train', 'val', 'test'),
                   percentages=(0.6, 0.3, 0.1),
                   shuffle=True):
    """
    Divide samples into several sets (train, val, test), and save them into some text files
    :param file_dir: str, directory of all samples
    :param save_dir: str, directory for saving divided results
    :param name: str, custom name
    :param types: tuple, types of sample set ready to divide
    :param percentages: tuple, percentages corresponding types above
    :param shuffle: bool, shuffle or not shuffle the original samples
    """
    if not os.path.isdir(file_dir):
        raise NotADirectoryError('Please check out the directory: %s'
                                 % file_dir)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
        print('Create a new directory: %s' % save_dir)

    for root, dirs, files in os.walk(file_dir):
        if shuffle:
            random.shuffle(files)
            print('Shuffled')

        count = len(files)
        nums = []
        for i in range(len(percentages)-1):
            nums.append(int(percentages[i]*count))
        nums.append(count-sum(nums))
        print('Total: %d' % count)
        print('Types:', types)
        print('Numbers:', nums)

        idx = 0
        for type_i in range(len(nums)):
            file_name = os.path.join(save_dir, '%s_%s.txt' % (name, types[type_i]))
            end_idx = idx+nums[type_i]
            with open(file_name, 'w') as f:
                while idx < end_idx:
                    sample_name = files[idx].split('.')[0]
                    f.write('%s\n' % sample_name)
                    idx += 1


if __name__ == '__main__':
    f_dir = r'D:\Documents\Study\Data\RSOD-Dataset\playground\Annotation\xml'
    s_dir = r'D:\Documents\Study\Data\RSOD-Dataset\ImageSets\Main'
    name = 'playground'
    divide_samples(f_dir, s_dir, name)