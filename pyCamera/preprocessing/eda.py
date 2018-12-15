#encoding：utf-8
# 类别分布
from collections import Counter
def print_distribution(files, classes=None):
    if classes is None:
        classes = [p.split("/")[-2] for p in files]
    classes_count = Counter(classes)
    for class_name, class_count in classes_count.items():
        print('{:>22}: {:5d} ({:04.1f}%)'.format(class_name, class_count, 100. * class_count / len(classes)))