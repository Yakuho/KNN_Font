# -*- coding: utf-8 -*-
# Author: Yakuho
# Date  : 2019/11/16
from xml.dom.minidom import parse
from fontTools.ttLib import TTFont
from dict_font import font_dict, font_chi
import numpy as np
import KNN


def save_xml(filename):
    font = TTFont(f'{filename}.woff')
    font.saveXML(f'{filename}.xml')


def get_offset_font(filename):
    data = parse(filename)
    collection = data.documentElement
    labels = collection.getElementsByTagName("TTGlyph")
    data_list = []
    max_len = 0
    for label in labels:
        contour = label.getElementsByTagName("contour")
        offset = [[label.getAttribute("name"),
                          label.getAttribute("yMax"),
                          label.getAttribute("yMin"),
                          label.getAttribute("xMax"),
                          label.getAttribute("xMin")]]
        for item in contour:
            pt = item.getElementsByTagName("pt")
            for xy in pt:
                if xy.hasAttribute("y"):
                    offset.append(int(xy.getAttribute("y")))
                if xy.hasAttribute("x"):
                    offset.append(int(xy.getAttribute("x")))
        else:
            data_list.append(offset)
            max_len = max_len if max_len > len(offset) else len(offset)
    for i in range(603):
        data_list[i] = data_list[i] + [0]*(max_len-len(data_list[i]))
    return data_list


def get_label_font(labels_np: list or np.ndarray):
    label_list = []
    for item in labels_np:
        label_list.append(item[0][0])
    return label_list


# 训练用
data_list1 = get_offset_font('3aefeca3.xml')
group = np.array(data_list1)[:, 1:].tolist()
labels = get_label_font(np.array(data_list1)[:, :1])
normalize_group = KNN.normalize_data_z_score_arctan(group)
# 测试用
data_list2 = get_offset_font('892bb594.xml')
test_group = np.array(data_list2)[:, 1:].tolist()
test_labels = get_label_font(np.array(data_list2)[:, :1])
normalize_test_group = KNN.normalize_data_z_score_arctan(test_group)
# 测试字体的映射
test_unicode_list = [TTFont('892bb594.woff').getGlyphName(a) for a in range(603)]
test_dict = {i: j for i, j in zip(test_unicode_list, font_chi)}

i ,j = 0, 0
total = len(test_labels)
for item in np.array(test_group).tolist():
    result = KNN.classify_knn(item, dataSet=normalize_group, labels=labels, k=3)
    result_fact = test_dict[test_labels[i]]
    result = font_dict[result]
    print(f'正在预测中，预测结果为【{result}】 实际结果为【{result_fact}】.')
    i += 1
    if result == result_fact:
        j += 1
    else:
        print('↑')
else:
    print(f'准确率为{round(j / total * 100, 2)}%')