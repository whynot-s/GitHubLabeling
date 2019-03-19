import tensorflow as tf
import gensim


class TextCNN:
    def __init__(self, filter_sizes, num_filters, num_classes, learning_rate, batch_size, decay_steps, decay_rate, sequence_length,
                 embed_size, correct_threshold, word2vec_model_path, is_training,
                 initializer=tf.random_normal_initializer(stddev=0.1), clip_gradients=5.0, decay_rate_big=0.5):
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.embed_size = embed_size
        self.learning_rate = tf.Variable(learning_rate, trainable=False, name="learning_rate")
        self.learning_rate_decay_half_op = tf.assign(self.learning_rate, self.learning_rate * decay_rate_big)
        self.num_filters = num_filters
        self.initializer = initializer
        self.filter_sizes = filter_sizes
        self.num_filters_total = self.num_filters * len(filter_sizes)
        self.clip_gradients = clip_gradients
        self.correct_threshold = correct_threshold
        self.is_training_flag = is_training
        self.w2vModel = gensim.models.Word2Vec.load(word2vec_model_path)

        self.input_x = tf.placeholder(tf.float32, [None, self.sequence_length, self.embed_size], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, self.num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.iter = tf.placeholder(tf.int32)
        self.tst = tf.placeholder(tf.bool)
        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))
        self.b1 = tf.Variable(tf.ones([self.num_filters]) / 10)
        self.b2 = tf.Variable(tf.ones([self.num_filters]) / 10)
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate

        self.instantiate_weights()
        self.logits = self.inference()
        self.loss_val = self.loss()
        self.train_op = self.train()
        # precision, recall
        pred = tf.nn.sigmoid(self.logits)
        label_true = tf.count_nonzero(self.input_y, axis=1)
        logit_temp = tf.where(pred >= self.correct_threshold, x=pred, y=tf.zeros(tf.shape(pred)))
        logit_true = tf.cast(tf.count_nonzero(logit_temp, axis=1), tf.float32)
        logit_true_bz = tf.where(logit_true < 1, x=tf.zeros(tf.shape(logit_true)) + 1, y=logit_true)
        pred_correct = tf.count_nonzero(tf.multiply(tf.cast(self.input_y, tf.float32), logit_temp), axis=1)
        self.precision = tf.reduce_mean(tf.divide(tf.cast(pred_correct, tf.float32), logit_true_bz))
        self.recall = tf.reduce_mean(tf.divide(pred_correct, label_true))

        tf.summary.scalar('loss', self.loss_val)
        tf.summary.scalar('precision', self.precision)
        tf.summary.scalar('recall', self.recall)

    def instantiate_weights(self):
            with tf.name_scope("embedding"):
                self.W_projection = tf.get_variable("W_projection", shape=[self.num_filters_total, self.num_classes],
                                                    initializer=self.initializer)
                self.b_projection = tf.get_variable("b_projection", shape=[self.num_classes])

    def inference(self):
        self.embedded_words = self.input_x
        self.sentence_embeddings_expanded = tf.expand_dims(self.embedded_words, -1)
        h = self.cnn_multiple_layers()
        with tf.name_scope("output"):
            logits = tf.matmul(h, self.W_projection) + self.b_projection
        return logits

    def cnn_multiple_layers(self):
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.variable_scope('cnn_multiple_layers' + "convolution-pooling-%s" % filter_size):
                filter = tf.get_variable("filter-%s" % filter_size, [filter_size, self.embed_size, 1, self.num_filters], initializer=self.initializer)
                conv = tf.nn.conv2d(self.sentence_embeddings_expanded, filter, strides=[1, 1, 1, 1], padding="VALID", name="conv")
                # shape:[batch_size,sequence_length - filter_size + 1,1,num_filters]
                conv = tf.contrib.layers.batch_norm(conv, is_training=self.is_training_flag, scope='cnn1')
                b = tf.get_variable("b-%s" % filter_size, [self.num_filters])
                h = tf.nn.relu(tf.nn.bias_add(conv, b), "relu")
                h = tf.reshape(h, [-1, self.sequence_length - filter_size + 1, self.num_filters, 1])
                # shape:[batch_size,sequence_length - filter_size + 1,num_filters,1]
                filter2 = tf.get_variable("filter2-%s" % filter_size, [filter_size, self.num_filters, 1, self.num_filters], initializer=self.initializer)
                conv2 = tf.nn.conv2d(h, filter2, strides=[1, 1, 1, 1], padding="VALID", name="conv2")
                # shape:[batch_size,sequence_length - filter_size * 2 + 2,1,num_filters]
                conv2 = tf.contrib.layers.batch_norm(conv2, is_training=self.is_training_flag, scope='cnn2')
                b2 = tf.get_variable("b2-%s" % filter_size, [self.num_filters])
                h = tf.nn.relu(tf.nn.bias_add(conv2, b2), "relu2")
                # shape:[batch_size,sequence_length - filter_size * 2 + 2,1,num_filters]
                pooling_max = tf.squeeze(tf.nn.max_pool(h, ksize=[1, self.sequence_length - filter_size * 2 + 2, 1, 1], strides=[1, 1, 1, 1], padding='VALID', name="pool"))
                pooled_outputs.append(pooling_max)  # h:[batch_size,num_filters]
        h = tf.concat(pooled_outputs, axis=1)  # [batch_size,num_filters*len(self.filter_sizes)]
        with tf.name_scope("dropout"):
            h = tf.nn.dropout(h, keep_prob=self.dropout_keep_prob)  # [batch_size,num_filters*len(self.filter_sizes)]
        return h  # [batch_size,num_filters*len(self.filter_sizes)]

    def loss(self, l2_lambda=0.00001):
        with tf.name_scope("loss"):
            losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(self.input_y, tf.float32), logits=self.logits)
            losses = tf.reduce_sum(losses, axis=1)
            loss = tf.reduce_mean(losses)
            l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            loss = loss + l2_losses
        return loss

    def train(self):
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps, self.decay_rate, staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        gradients, variables = zip(*optimizer.compute_gradients(self.loss_val))
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.apply_gradients(zip(gradients, variables))
        return train_op
