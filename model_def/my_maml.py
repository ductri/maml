import logging
import tensorflow as tf
import matplotlib.pyplot as plt
import time

from naruto_skills import graph_utils
from model_def.data_generator import DataGenerator
from model_def.sine_model import SineModel

logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    with tf.Graph().as_default() as gr:
        no_task, no_sample, input_dim, output_dim = 32, None, 1, 1
        meta_lr = 1e-3

        tf_X_train = tf.placeholder(dtype=tf.float32, shape=(no_task, no_sample, input_dim), name='X_train')
        tf_y_train = tf.placeholder(dtype=tf.float32, shape=(no_task, no_sample, output_dim), name='y_train')
        tf_X_eval = tf.placeholder(dtype=tf.float32, shape=(no_task, no_sample, input_dim), name='X_eval')
        tf_y_eval = tf.placeholder(dtype=tf.float32, shape=(no_task, no_sample, output_dim), name='y_eval')

        sine_model = SineModel(input_dim, output_dim)

        tf_eval_losses = []
        tf_train_losses = []  # Just for monitoring
        tf_list_logits = []

        for task_idx in range(no_task):

            tf_train_logits, tf_train_loss = sine_model.do_it(tf_X_train[task_idx], tf_y_train[task_idx], sine_model.meta_weights)
            tf_list_logits.append(tf_train_logits)  # for predict after meta-training and task training

            grads = tf.gradients(tf_train_loss, [sine_model.meta_weights[k] for k in sine_model.meta_weights.keys()])
            grads = dict(zip(sine_model.meta_weights.keys(), grads))
            task_weights = {k: (v - meta_lr * grads[k]) for k, v in sine_model.meta_weights.items()}

            _, tf_train_loss = sine_model.do_it(tf_X_train[task_idx], tf_y_train[task_idx], task_weights)
            grads = tf.gradients(tf_train_loss, [task_weights[k] for k in task_weights.keys()])
            grads = dict(zip(task_weights.keys(), grads))
            task_weights = {k: v - meta_lr * grads[k] for k, v in task_weights.items()}

            tf_eval_logits, tf_eval_loss = sine_model.do_it(tf_X_eval[task_idx], tf_y_eval[task_idx], task_weights)
            _, tf_train_loss = sine_model.do_it(tf_X_train[task_idx], tf_y_train[task_idx], task_weights)

            tf_train_losses.append(tf_train_loss)

            tf_eval_losses.append(tf_eval_loss)

        tf_mean_eval_loss = tf.reduce_mean(tf.stack(tf_eval_losses, axis=0), axis=0)
        tf_global_step = tf.get_variable(name='global_step', dtype=tf.int32, initializer=0)
        tf_task_global_step = tf.get_variable(name='task_global_step', dtype=tf.int32, initializer=0)
        tf_optimizer_op = tf.train.AdamOptimizer(meta_lr).minimize(tf_mean_eval_loss, global_step=tf_global_step)
        tf_task_optimizer_op = tf.train.AdamOptimizer(meta_lr).minimize(tf_train_losses[0],
                                                                        global_step=tf_task_global_step)

        tf_mean_train_loss = tf.reduce_mean(tf.stack(tf_train_losses, axis=0), axis=0)

        writer = tf.summary.FileWriter(logdir='log/', graph=gr)
        writer.flush()
        writer.close()

        logging.info('Total vars: %s' % graph_utils.count_trainable_variables())

        data_generator = DataGenerator(num_samples_per_task=10, batch_size=32)

        with tf.Session().as_default() as sess:
            sess.run(tf.global_variables_initializer())
            start = time.time()
            for _ in range(10000):
                inputs, outputs, _, _ = data_generator.generate_sinusoid_batch()
                input_train, input_eval, output_train, output_eval = inputs[:, :5], inputs[:, 5:], outputs[:,
                                                                                                   :5], outputs[:, 5:]
                _, global_step = sess.run([tf_optimizer_op, tf_global_step],
                                          feed_dict={tf_X_train: input_train, tf_X_eval: input_eval,
                                                     tf_y_train: output_train, tf_y_eval: output_eval})
                if global_step % 50 == 0 or global_step == 1:
                    mean_eval_loss, mean_train_loss = sess.run([tf_mean_eval_loss, tf_mean_train_loss],
                                                               feed_dict={tf_X_train: input_train,
                                                                          tf_X_eval: input_eval,
                                                                          tf_y_train: output_train,
                                                                          tf_y_eval: output_eval})
                    logging.info('Step: %s\tMean train loss: %s \tMean eval loss: %s\t Duration: %.2f s', global_step,
                                 mean_train_loss, mean_eval_loss, time.time() - start)
                    start = time.time()

        logging.info('Training done. Evaluate on novel task')
        logging.info('\n')
        # Actually, we're training for the last task
        data_generator = DataGenerator(num_samples_per_task=100, batch_size=32)
        inputs, outputs, _, _ = data_generator.generate_sinusoid_batch()
        train_size = 5
        input_train, input_eval, output_train, output_eval = inputs[:, :train_size], inputs[:, train_size:], \
                                                             outputs[:, :train_size], outputs[:, train_size:]
        task_idx = 0
        task_global_step, task_train_loss, task_eval_loss = sess.run(
            [tf_task_global_step, tf_train_losses[task_idx], tf_eval_losses[task_idx]],
            feed_dict={tf_X_train: input_train, tf_y_train: output_train, tf_X_eval: input_eval,
                       tf_y_eval: output_eval})
        logging.info('Step: %s\t Train loss: %.5f\t Eval loss: %.5f', task_global_step, task_train_loss, task_eval_loss)
        # Update gradient exactly xx times
        for _ in range(500):
            _, task_global_step, task_train_loss, task_eval_loss = sess.run(
                [tf_task_optimizer_op, tf_task_global_step, tf_train_losses[task_idx], tf_eval_losses[task_idx]],
                feed_dict={tf_X_train: input_train, tf_y_train: output_train, tf_X_eval: input_eval,
                           tf_y_eval: output_eval})
            logging.info('Step: %s\t Train loss: %.5f\t Eval loss: %.5f', task_global_step, task_train_loss,
                         task_eval_loss)

        plt.figure()
        predict = sess.run(tf_list_logits[task_idx], feed_dict={tf_X_train: input_eval})
        plt.plot(input_eval[0], predict, 'ro')
        plt.plot(input_eval[0], output_eval[0], 'go')
        plt.show()
