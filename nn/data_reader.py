import argparse
from bs4 import BeautifulSoup
import re
import xml.sax.saxutils as saxutils
import codecs
import os


NUMBER_OF_FILES = 22


def strip_tags(text):
    return re.sub('<[^<]+?>', '', text).strip()


def parse_reuters():
    """
    Parse all files removing xml tags
    :return: train and test data
    """
    _train_text_lst, _train_label_lst, _test_text_lst, _test_label_lst = [], [], [], [] 
    for idx in range(NUMBER_OF_FILES):
        file_path = os.path.join('..', 'data', 'reut2-0{}.sgm'.format('0{}'.format(idx) if idx < 10 else idx))
        print('processing file {}'.format(file_path))
        with codecs.open(file_path, 'r', encoding='UTF8', errors='replace') as f:
            content = BeautifulSoup(f.read().lower(), 'html.parser')
            for doc in content('reuters'):
                doc_split = doc['lewissplit']
                doc_topics = [strip_tags(str(topic)) for topic in doc.topics.contents]
                if not doc_topics:
                    continue
                doc_body = saxutils.unescape(strip_tags(str(doc('text')[0].body)).replace('reuter\n&#3;', ''))
                if doc_split == 'train':
                    _train_text_lst.append(doc_body); _train_label_lst.append(doc_topics)
                elif doc_split == 'test':
                    _test_text_lst.append(doc_body); _test_label_lst.append(doc_topics)
    return _train_text_lst, _train_label_lst, _test_text_lst, _test_label_lst


def topics_with_occurences_gt(threshold, _train_label_lst):
    """
    Filter out topics that has occurrences greater than threshold
    :param threshold:
    :return: list of remaining topics
    """
    with open(os.path.join('..', 'data', 'all-topics-strings.lc.txt')) as t_file:
        topic_lst = [category.strip().lower() for category in t_file.readlines()]
    train_topic_count = {topic: 0 for topic in topic_lst}
    for labels in _train_label_lst:   # try with test_label_lst
        for t in labels:
            train_topic_count[t] += 1
    topics = [key for key in train_topic_count if train_topic_count[key] > threshold]
    print('Removed %d topics with occurrences greater than %d. remain: %d topics' % (len(topic_lst) - len(topics),
                                                                                     threshold, len(topics)))
    return topics


def remove_minor_classes(topics, _train_text_lst, _train_label_lst, _test_text_lst, _test_label_lst):
    """
    remove samples with little occurrences topics
    :param topics: topics after removal
    :return: train and test data after removal
    """
    print('Removing samples with minor topics')
    def remove_minor_class(_label_lst):
        label_lst = _label_lst.copy()
        idx_selected = []
        for i, topics_sample in enumerate(label_lst):
            intersect = list(set(topics_sample) & set(topics))
            if len(intersect) > 0:
                idx_selected.append(i)
                label_lst[i] = intersect
        return idx_selected, label_lst

    # remove samples with few class occurrences in training set
    train_idx_selected, train_label_lst = remove_minor_class(_train_label_lst)
    train_text_lst = [_train_text_lst[i] for i in train_idx_selected]
    train_label_lst = [train_label_lst[i] for i in train_idx_selected]

    # remove samples with few class occurrences in test set
    test_idx_selected, test_label_lst = remove_minor_class(_test_label_lst)
    test_text_lst = [_test_text_lst[i] for i in test_idx_selected]
    test_label_lst = [test_label_lst[i] for i in test_idx_selected]
    print('Train samples: %d' % len(train_label_lst))
    print('Test samples: %d' % len(test_label_lst))
    return train_text_lst, train_label_lst, test_text_lst, test_label_lst
