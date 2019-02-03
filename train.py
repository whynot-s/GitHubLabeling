import tensorflow as tf
from textRNN import TextRNN
import numpy as np

correct_threshold = 0.5
label_size = 1954
embedding_size = 200
context_size = 128
sequence_length = 200

learning_rate = 0.001
training_iters = 10000
display_step = 1
batch_size = 100

cfg = tf.ConfigProto(
    device_count={"CPU": 4},
    inter_op_parallelism_threads = 0,
    intra_op_parallelism_threads = 0,
    log_device_placement=True
)

with tf.Session() as sess:
    textrnn = TextRNN(
        batch_size=batch_size,
        embedding_size=embedding_size,
        context_size=context_size,
        classes_num=label_size,
        correct_threshold=correct_threshold,
        sequence_length=sequence_length,
        dropout_keep_prob=1
    )
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(textrnn.loss)
    sess.run(tf.global_variables_initializer())

    step = 0
    loss_total = 0
    prec_total = 0
    reca_total = 0

    import trainingData

    while step < training_iters:
        X, Y = trainingData.next_batch(sequence_length, label_size, batch_size=batch_size)
        init_state = np.zeros([batch_size, context_size])
        # X, Y = trainingData.mock(batch_size, sequence_length, embedding_size, label_size)
        _, prec, reca, lossval = sess.run([optimizer, textrnn.precision, textrnn.recall, textrnn.loss],
                                                   feed_dict={ textrnn.input_X: X, textrnn.input_Y: Y,
                                                               textrnn.c_left: init_state, textrnn.c_right: init_state})
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
        step += 1