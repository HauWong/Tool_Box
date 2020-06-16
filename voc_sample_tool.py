# !/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import random
from xml.dom.minidom import Document


class MyDoc(Document):
    def __init__(self):
        super(MyDoc, self).__init__()
        
    def add_one_text_node(self, parent, name, content):
        info = self.createElement(name)
        info_content = self.createTextNode(str(content))
        info.appendChild(info_content)
        parent.appendChild(info)


def shuffle_samples(sample_ls):
    res = sample_ls
    return res


def write_xml(image_info, boxes, save_path):
    """
    Write annotation information to a xml file in a VOC-format
    :param image_info: dict, image information,
                       {'folder':'',
                        'filename':'',
                        'size': (w, h, c),
                        'segmented': True or False,
                        'source':[optional]}
    :param boxes: list, list of boxes,
                     [{'name':'',
                       'bndbox': (xmin, ymin, xmax, ymax),
                       'truncated': True or False,
                       'difficult': True or False}, ...]
    :param save_path: path for saving xml file
    """
    doc = MyDoc()
    anno = doc.createElement('annotation')
    doc.appendChild(anno)
    for k, v in image_info.items():
        if k == 'size':
            size = doc.createElement('size')
            anno.appendChild(size)
            for i, t in enumerate(['width', 'height', 'depth']):
                doc.add_one_text_node(size, t, v[i])
        elif k == 'segmented':
            doc.add_one_text_node(anno, k, 1 if v else 0)
        elif k == 'source':
            source = doc.createElement('source')
            anno.appendChild(source)
            for s_k, s_v in v.items():
                doc.add_one_text_node(source, s_k, s_v)
        else:
            doc.add_one_text_node(anno, k, v)

    for box in boxes:
        obj = doc.createElement('object')
        anno.appendChild(obj)
        for k, v in box.items():
            if k == 'name':
                doc.add_one_text_node(obj, k, v)
            elif k == 'bndbox':
                bndbox = doc.createElement('bndbox')
                obj.appendChild(bndbox)
                for i, t in enumerate(['xmin', 'ymin', 'xmax', 'ymax']):
                    doc.add_one_text_node(bndbox, t, v[i])
            elif k == 'truncated' or k == 'difficult':
                doc.add_one_text_node(obj, k, 1 if v else 0)
            else:
                print('Warning: element %f not written.' % k)

    with open(save_path, 'wb') as f:
        f.write(doc.toprettyxml(indent='\t', encoding='utf-8'))



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
    # divide_samples(f_dir, s_dir, name)
    image_info = {'folder':'',
                        'filename':'',
                        'size': (200, 200, 3),
                        'segmented':0}
    boxes = [{'name':'a',
                       'bndbox': (0, 0, 50, 50),
                       'truncated': False,
                       'difficult': False},
             {'name':'b',
                       'bndbox': (10, 10, 60, 80),
                       'truncated': False,
                       'difficult': False},
             {'name': 'b',
              'bndbox': (20, 10, 50, 70),
              'truncated': False,
              'difficult': False}]
    save_path = r'D:\PCFiles\Desktop\test.xml'
    write_xml(image_info, boxes, save_path)
