# coding: utf-8

# 引用部分
from __future__ import print_function
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import tensorflow as tf

# 主函数


def main(_):
    # 从csv中读取数据
    csv_file_name = './grove.csv'
    reader = tf.contrib.timeseries.CSVReader(csv_file_name)
    # 一个batch内共有batch_size个序列，每个序列的长度为window_size。
    train_input_fn = tf.contrib.timeseries.RandomWindowInputFn(
        reader, batch_size=32, window_size=16)
    with tf.Session() as sess:
        data = reader.read_full()
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(sess=sess, coord=coord)
        data = sess.run(data)
        coord.request_stop()

    # 自回归模型
    # periodicities延迟期数
    ar = tf.contrib.timeseries.ARRegressor(
        periodicities=20, input_window_size=10, output_window_size=6,
        num_features=1,
        loss=tf.contrib.timeseries.ARModel.NORMAL_LIKELIHOOD_LOSS)

    ar.train(input_fn=train_input_fn, steps=50)

    evaluation_input_fn = tf.contrib.timeseries.WholeDatasetInputFn(reader)

    evaluation = ar.evaluate(input_fn=evaluation_input_fn, steps=1)

    # 这里的step是预测期数
    (predictions,) = tuple(ar.predict(
        input_fn=tf.contrib.timeseries.predict_continuation_input_fn(
            evaluation, steps=10)))

    # 画图
    plt.figure(figsize=(15, 5))
    plt.plot(data['times'].reshape(-1),
             data['values'].reshape(-1), label='origin')
    plt.plot(evaluation['times'].reshape(-1),
             evaluation['mean'].reshape(-1), label='evaluation')
    plt.plot(predictions['times'].reshape(-1),
             predictions['mean'].reshape(-1), label='prediction')
    plt.xlabel('time_step')
    plt.ylabel('values')
    plt.legend(loc=4)
    plt.savefig('ar_result.jpg')


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
