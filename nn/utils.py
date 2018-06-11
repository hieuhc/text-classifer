import numpy as np


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """Iterate the data batch by batch"""
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(data_size / batch_size) + 1

    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def get_label_using_scores_by_threshold(scores, threshold=0.5):
    """
    Get the predicted labels based on the threshold.
    If there is no predict value greater than threshold, then choose the label which has the max predict value.
    """
    predicted_labels = []
    predicted_values = []
    y_pred = np.zeros(scores.shape)
    scores = np.ndarray.tolist(scores)
    for score in scores:
        count = 0
        index_list = []
        value_list = []
        for index, predict_value in enumerate(score):
            if predict_value > threshold:
                index_list.append(index)
                value_list.append(predict_value)
                count += 1
        if count == 0:
            index_list.append(score.index(max(score)))
            value_list.append(max(score))
        predicted_labels.append(index_list)
        predicted_values.append(value_list)
    for row in range(y_pred.shape[0]):
        y_pred[row, predicted_labels[row]] = 1
    return predicted_labels, predicted_values, y_pred


def cal_metric(predicted_labels, labels):
    """
    Calculate the metric(recall, accuracy).
    """
    label_no_zero = []
    for index, label in enumerate(labels):
        if int(label) == 1:
            label_no_zero.append(index)
    count = 0
    for predicted_label in predicted_labels:
        if int(predicted_label) in label_no_zero:
            count += 1
    rec = count / len(label_no_zero)
    acc = count / len(predicted_labels)
    return rec, acc


def cal_F(rec, acc):
    """
    Calculate the metric F value.
    """
    if (rec + acc) == 0:
        F = 0.0
    else:
        F = (2 * rec * acc) / (rec + acc)
    return F
