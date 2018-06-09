import os
import tensorflow as tf
from sklearn import preprocessing
from tensorflow.contrib import learn
from sklearn.model_selection import train_test_split
import numpy as np
import text_cnn
import utils
from data_reader import parse_reuters, remove_minor_classes, topics_with_occurences_gt
import yaml

# read data from file
_train_text_lst, _train_label_lst, _test_text_lst, _test_label_lst = parse_reuters()

# remove classes with very few occurrences
topics_selected = topics_with_occurences_gt(3, _train_label_lst)
train_text_lst, train_label_lst, test_text_lst, test_label_lst = remove_minor_classes(topics_selected, _train_text_lst, _train_label_lst, _test_text_lst, _test_label_lst)

# label transformer
label_binarizer = preprocessing.MultiLabelBinarizer(topics_selected)
label_binarizer.fit(train_label_lst)
y_train = label_binarizer.transform(train_label_lst)
y_test = label_binarizer.transform(test_label_lst)

# transform words to words index using vocab_processor
max_document_length = max([len(x.split(' ')) for x in train_text_lst])
print('The maximum length of all sentences: {}'.format(max_document_length))
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x_train = np.array(list(vocab_processor.fit_transform(train_text_lst)))
x_test = np.array(list(vocab_processor.transform(test_text_lst)))
print('x train shape: {}'.format(x_train.shape))
print('x test shape: {}'.format(x_test.shape))


# split dev set for evaluation during train
shuffle_indices = np.random.permutation(np.arange(len(y_train)))
x_shuffled = x_train[shuffle_indices]
y_shuffled = y_train[shuffle_indices]
x_train, x_dev, y_train, y_dev = train_test_split(x_shuffled, y_shuffled, test_size=0.1)


# initialize tf graph and session
with open('config.yaml', 'r') as cf:
    config = yaml.load(cf)
graph = tf.Graph()
session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
sess = tf.Session(config=session_conf)
cnn = text_cnn.TextCNN(
    sequence_length=x_train.shape[1],
    num_classes=y_train.shape[1],
    vocab_size=len(vocab_processor.vocabulary_),
    embedding_size=config['embedding_size'],
    filter_sizes=config['filter_size'],
    num_filters=config['num_filters'],
    l2_reg_lambda=config['l2_reg_lambda'])
global_step = tf.Variable(0, name="global_step", trainable=False)
learning_rate = tf.train.exponential_decay(learning_rate=config['learning_rate'],
                                           global_step=global_step,
                                           decay_steps=config['decay_steps'],
                                           decay_rate=config['decay_rate'], staircase=True)
optimizer = tf.train.AdamOptimizer(learning_rate)
grads_and_vars = optimizer.compute_gradients(cnn.loss)
train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
saver = tf.train.Saver()


#  define training and validation step
def train_step(x_batch, y_batch):
    feed_dict = {
        cnn.input_x: x_batch,
        cnn.input_y: y_batch,
        cnn.dropout_keep_prob: config['dropout_keep_prob']}
    _, step, loss = sess.run([train_op, global_step, cnn.loss], feed_dict)
    return loss


def dev_step(x_batch, y_batch):
    feed_dict = {cnn.input_x: x_batch, cnn.input_y: y_batch, cnn.dropout_keep_prob: 1.0}
    step, scores, loss = sess.run([global_step, cnn.scores, cnn.loss], feed_dict)
    predicted_labels_threshold, predicted_values_threshold, y_batch_pred = utils.get_label_using_scores_by_threshold(scores=scores, threshold=0.5)

    cur_rec_ts, cur_acc_ts, cur_F_ts = 0.0, 0.0, 0.0

    for index, predicted_label_threshold in enumerate(predicted_labels_threshold):
        rec_inc_ts, acc_inc_ts = utils.cal_metric(predicted_label_threshold, y_batch[index])
        cur_rec_ts, cur_acc_ts = cur_rec_ts + rec_inc_ts, cur_acc_ts + acc_inc_ts

    cur_rec_ts /= len(y_batch)
    cur_acc_ts /= len(y_batch)
    cur_F_ts = utils.cal_F(cur_rec_ts, cur_acc_ts)
    return cur_rec_ts, cur_acc_ts, cur_F_ts, loss, y_batch_pred


# start training process
sess.run(tf.global_variables_initializer())

# train the cnn model with x_train and y_train (batch by batch)
train_batches = utils.batch_iter(list(zip(x_train, y_train)), config['batch_size'], config['num_epoches'])
best_accuracy, best_at_step = 0, 0
num_batches_per_epoch = int((len(x_train) - 1) / config['batch_size']) + 1
for train_batch in train_batches:
    x_train_batch, y_train_batch = zip(*train_batch)
    train_loss = train_step(x_train_batch, y_train_batch)
    current_step = tf.train.global_step(sess, global_step)
    if current_step % 50 == 0:
        print('Train step %d loss: %.4f' % (current_step, train_loss))

    # evaluate the model with x_dev and y_dev (batch by batch)
    if current_step % config['evaluate_every'] == 0:
        print('---------- Evaluate on step %d -----------' % current_step)
        dev_batches = utils.batch_iter(list(zip(x_dev, y_dev)), 32, 1)
        eval_counter, eval_loss, eval_rec_ts, eval_acc_ts, eval_F_ts = 0, 0.0, 0.0, 0.0, 0.0
        for dev_batch in dev_batches:
            x_dev_batch, y_dev_batch = zip(*dev_batch)
            cur_rec_ts, cur_acc_ts, cur_F_ts, cur_loss, _ = dev_step(x_dev_batch, y_dev_batch)
            # update ts scores
            eval_counter += 1
            eval_rec_ts, eval_acc_ts = eval_rec_ts + cur_rec_ts, eval_acc_ts + cur_acc_ts
            eval_loss += cur_loss
        # calculate metrics on the whole dev set
        eval_loss = float(eval_loss / eval_counter)
        eval_rec_ts = float(eval_rec_ts / eval_counter)
        eval_acc_ts = float(eval_acc_ts / eval_counter)
        eval_F_ts = utils.cal_F(eval_rec_ts, eval_acc_ts)

        print('loss on dev set: {}'.format(eval_loss))
        print('recall on dev set: {}'.format(eval_rec_ts))
        print('accuracy on dev set: {}'.format(eval_acc_ts))
        print('F score on dev set: {}'.format(eval_F_ts))

    # save the model if it is the best based on accuracy on dev set
    if current_step % config['checkpoint_every'] == 0:
        checkpoint_prefix = os.path.join("checkpoint", "model")
        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
        print('Saved model to {}'.format(path))
    if current_step % num_batches_per_epoch == 0:
        current_epoch = current_step // num_batches_per_epoch
        print(" Epoch {} has finished!".format(current_epoch))


# predict test data
from sklearn import metrics
test_batches = utils.batch_iter(list(zip(x_test, y_test)), batch_size=32, num_epochs=1)
y_preds, y_test_reform = [], []
for test_batch in test_batches:
    x_test_batch, y_test_batch = zip(*test_batch)
    _, _, _, _, y_pred_batch = dev_step(x_test_batch, y_test_batch)
    y_preds.append(y_pred_batch)
    y_test_reform.append(y_test_batch)
y_preds = np.vstack(y_preds)
y_test_reform = np.vstack(y_test_reform)
report = metrics.classification_report(y_test_reform, y_preds, target_names=topics_selected)
print(report)
with(open(os.path.join(os.getcwd(), 'result.txt'), 'w')) as f:
    f.write(report)