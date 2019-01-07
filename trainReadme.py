import tensorflow as tf
from tensorflow.contrib import rnn
import time

start_time = time.time()

def elapsed(sec):
    if sec < 60:
        return str(sec) + " sec"
    elif sec < 60 * 60:
        return str(sec / 60) + " min"
    else:
        return str(sec / (60 * 60)) + " hr"


label_size = 3400
word_vector_dim = 200

learning_rate = 0.001
training_iters = 10000
display_step = 1000
batch_size = 20
n_input = 6

n_hidden1 = 256
n_hidden2 = 512
n_hidden3 = 512

x = tf.placeholder(tf.float32, [batch_size, n_input, word_vector_dim])
label = tf.placeholder(tf.int32, [batch_size, label_size])

x1 = tf.reshape(x, [-1, n_input, word_vector_dim])
x2 = tf.split(x1, n_input, word_vector_dim)
rnn_cell = rnn.MultiRNNCell([rnn.LSTMCell(n_hidden1), rnn.LSTMCell(n_hidden2), rnn.LSTMCell(n_hidden3)])
outputs, states = rnn.static_rnn(rnn_cell, x2, dtype=tf.float32)

pred = tf.contrib.layers.fully_connected(outputs[-1], label_size, activation_fn=None)
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=label))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

label_true = tf.count_nonzero(label)
logit_temp = tf.where(pred >= 0.5, x=pred, y=tf.zeros(tf.shape(pred)))
logit_true = tf.count_nonzero(logit_temp)
pred_correct = tf.count_nonzero(tf.multiply(tf.cast(label, tf.float32), logit_temp))

precision = tf.divide(tf.cast(pred_correct, tf.float32), tf.cast(logit_true, tf.float32))
recall = tf.divide(tf.cast(pred_correct, tf.float32), tf.cast(label_true, tf.float32))

savedir = "log/readme_v1/"
saver = tf.train.Saver(max_to_keep=1)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    step = 0
    loss_total = 0
    prec_total = 0
    reca_total = 0

    kpt = tf.train.latest_checkpoint(savedir)
    print("kpt:", kpt)
    startepo = 0
    if kpt is True:
        saver.restore(sess, kpt)
        ind = kpt.find("-")
        startepo = int(kpt[ind+1:])
        print(startepo)
        step = startepo

    while step < training_iters:
        X = 0
        Y = 0

        _, prec, reca, lossval, predval = sess.run([optimizer, precision, recall, loss, pred], feed_dict={x: X, label: Y})
        loss_total += lossval
        prec_total += prec
        reca_total += reca
        if (step + 1) % display_step == 0:
            print("Iter= " + str(step + 1) + ", Average Loss= " +
                  "{:.6f}".format(loss_total / display_step) + ", Average Precision= " +
                  "{:.2f}%".format(100 * prec_total / display_step) + ", Average Recall= " +
                  "{:.2f}%".format(100 * reca_total / display_step))
            prec_total = 0
            reca_total = 0
            loss_total = 0
            saver.save(sess, savedir + "readme_v1.cpkt", global_step=step)
        step += 1

    print("Finished!")
    saver.save(sess, savedir + "readme_v1.cpkt", global_step=step)
    print("Elapsed time: ", elapsed(time.time() - start_time))