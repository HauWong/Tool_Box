# !/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import random
import numpy as np
import xml.etree.ElementTree as ET
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


def get_boxes_from_file(label_path):
    """
    Extract boxes from label file(*.xml)
    :param label_path: str, path of label file
    :return: list, boxes, [[xmin, ymin, xmax, ymax], ...]
    """
    res = []
    tree = ET.parse(label_path)
    for obj in tree.findall('object'):
        bbox = obj.find('bndbox')
        res.append([int(bbox.find('xmin').text),
                    int(bbox.find('ymin').text),
                    int(bbox.find('xmax').text),
                    int(bbox.find('ymax').text)])
    return res


def objects(boxes, names, truncated, difficult):
    """
    Generate voc format objects for annotation(*.xml)
    :param boxes: list, box list of all objects, [(row_min, row_max, col_min, col_max), (...), ...]
    :param names: list, names of all objects, ['waste', ...]
    :param truncated: list, is truncated or not, [False, ...]
    :param difficult: list, is difficult or not, [False, ...]
    :return: res, list, all objects, [{'name': ..., 'bndbox': ..., 'truncated': ..., 'difficult': ...}, ...]
    """

    res = []

    for i, box in enumerate(boxes):
        box_dict = {'name': names[i], 'bndbox': box, 'truncated': truncated[i], 'difficult': difficult[i]}
        res.append(box_dict)
    return res


def voc_to_coco(voc_lab_path, coco_lab_path, classes=('waste',)):
    """
    Convert voc_annotation_file to coco_label_file
    :param voc_lab_path: str, path of voc annotation
    :param coco_lab_path: str, path of coco label to save
    :param classes: tuple, all class names
    :return: None
    """
    root = ET.parse(voc_lab_path).getroot()
    img_w = int(root.find('size').find('width').text)
    img_h = int(root.find('size').find('height').text)

    with open(coco_lab_path, 'w') as f:
        for obj in root.iter('object'):
            try:
                cls_idx = classes.index(obj.find('name').text.lower().strip())
            except:
                raise ValueError('"%s" is not found in classes list!' %
                                 obj.find('name').text.lower().strip())

            bbox = obj.find('bndbox')
            x_min = int(bbox.find('xmin').text)
            y_min = int(bbox.find('ymin').text)
            x_max = int(bbox.find('xmax').text)
            y_max = int(bbox.find('ymax').text)

            x_center = (x_min+x_max)/(2*img_w)
            y_center = (y_min+y_max)/(2*img_h)
            b_width = (x_max-x_min)/img_w
            b_height = (y_max-y_min)/img_h

            s = '%d\t%0.6f\t%0.6f\t%0.6f\t%0.6f\n' %\
                (cls_idx, x_center, y_center, b_width, b_height)
            print(s)
            f.write(s)


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


# -------------------------------SSD代码迁移修改----------------------
def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        # obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text) - 1,
                              int(bbox.find('ymin').text) - 1,
                              int(bbox.find('xmax').text) - 1,
                              int(bbox.find('ymax').text) - 1]
        objects.append(obj_struct)

    return objects


def voc_ap(rec, prec, use_07_metric=True):
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(detpath, annopath, ovthresh=0.5, use_07_metric=True):

    recs = parse_rec(annopath)

    # extract gt objects for this class
    npos = 0
    R = [obj for obj in recs]
    bbox = np.array([x['bbox'] for x in R])
    difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
    det = [False] * len(R)
    npos = npos + sum(~difficult)
    class_recs = {'bbox': bbox,
                  'difficult': difficult,
                  'det': det}

    # read dets
    with open(detpath, 'r') as f:
        lines = f.readlines()
    if any(lines) == 1:

        splitlines = [x.strip().split(' ') for x in lines]
        image_ids = [x[0] for x in splitlines]
        confidence = np.array([float(x[1]) for x in splitlines])
        BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        ious = []
        for d in range(nd):
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            BBGT = class_recs['bbox'].astype(float)
            if BBGT.size > 0:
                # compute overlaps
                # intersection
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin, 0.)
                ih = np.maximum(iymax - iymin, 0.)
                inters = iw * ih
                uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                       (BBGT[:, 2] - BBGT[:, 0]) *
                       (BBGT[:, 3] - BBGT[:, 1]) - inters)
                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)
                ious.append(ovmax)

            if ovmax > ovthresh:
                if not class_recs['difficult'][jmax]:
                    if not class_recs['det'][jmax]:
                        tp[d] = 1.
                        class_recs['det'][jmax] = 1
                    else:
                        fp[d] = 1.
            else:
                fp[d] = 1.

        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, use_07_metric)
    else:
        rec = -1.
        prec = -1.
        ap = -1.
        ious = 0.

    return rec, prec, ap, ious


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
