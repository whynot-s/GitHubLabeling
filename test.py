import tensorflow as tf

batch_size = 5
embed_size = 4
num_filters = 2
sequence_length = 6

input_x = tf.placeholder(tf.float32, [batch_size, sequence_length, embed_size])
logits = tf.placeholder(tf.float32, [batch_size, 3])
input_y = tf.placeholder(tf.float32, [batch_size, 3])
correct_threshold = 0.5

pred = logits
label_true = tf.count_nonzero(input_y, axis=1)
logit_temp = tf.where(pred >= correct_threshold, x=pred, y=tf.zeros(tf.shape(pred)))
logit_true = tf.cast(tf.count_nonzero(logit_temp, axis=1), tf.float32)
logit_true_bz = tf.where(logit_true < 1, x=tf.zeros(tf.shape(logit_true))+1, y=logit_true)
pred_correct = tf.count_nonzero(tf.multiply(tf.cast(input_y, tf.float32), logit_temp), axis=1)
precision = tf.reduce_mean(tf.divide(tf.cast(pred_correct, tf.float32), logit_true_bz))
recall = tf.reduce_mean(tf.divide(pred_correct, label_true))

num_sentences = 2
embedded_words0 = tf.split(input_x, num_sentences, axis=1)
embedded_words0_shape = tf.shape(embedded_words0)
embedded_words = tf.stack(embedded_words0, axis=1)
embedded_words_shape = tf.shape(embedded_words)
embedded_words_reshaped = tf.reshape(embedded_words, shape=[-1, int(sequence_length / num_sentences), embed_size])
embedded_words_reshaped_shape = tf.shape(embedded_words_reshaped)

with tf.Session() as sess:
    logits_input = [[0.2, 0.3, 0.5], [0.2, 0.3, 0.5], [0.2, 0.3, 0.5], [0.6, 0.9, 0.1], [0.2, 0.3, 0.5]]
    y_ipnut = [[1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 1, 0], [0, 1, 1]]
    x_input = [[[2, 3, 4, 5], [1, 11, 111, 1111], [2, 3, 4, 5], [1, 11, 111, 1111], [2, 3, 4, 5], [1, 11, 111, 1111]],
               [[9, 99, 999, 9999], [5, 55, 555, 5555], [9, 99, 999, 9999], [5, 55, 555, 5555], [9, 99, 999, 9999], [5, 55, 555, 5555]],
               [[4, 44, 444, 4444], [6, 66, 666, 6666], [4, 44, 444, 4444], [6, 66, 666, 6666], [4, 44, 444, 4444], [6, 66, 666, 6666]],
               [[2, 3, 4, 5], [1, 11, 111, 1111], [2, 3, 4, 5], [1, 11, 111, 1111], [2, 3, 4, 5], [1, 11, 111, 1111]],
               [[2, 3, 4, 5], [1, 11, 111, 1111], [2, 3, 4, 5], [1, 11, 111, 1111], [2, 3, 4, 5], [1, 11, 111, 1111]]]
    print(sess.run(embedded_words0_shape, feed_dict={input_x:x_input}))
    print(sess.run(embedded_words_shape, feed_dict={input_x:x_input}))
    print(sess.run(embedded_words_reshaped_shape, feed_dict={input_x: x_input}))
    print(sess.run(embedded_words_reshaped, feed_dict={input_x: x_input}))

