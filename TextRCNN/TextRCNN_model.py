import tensorflow as tf
import copy


class TextRCNN:
    def __init__(self, num_classes, learning_rate, batch_size, decay_steps, decay_rate, sequence_length,
                 vocab_size, embed_size, is_training, correct_threshold, initializer=tf.random_normal_initializer(stddev=0.1)):
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.hidden_size = embed_size
        self.embed_size = embed_size
        self.correct_threshold = correct_threshold
        self.is_training = is_training
        self.initializer = initializer
        self.activation = tf.nn.tanh

        self.input_x = tf.placeholder(tf.float32, [None, self.sequence_length])
        self.input_y = tf.placeholder(tf.int32, [None, self.num_classes])
        self.dropout_keep_prob=tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))
        self.decay_steps, self.decay_rate = decay_steps, decay_rate

        self.instantiate_weights()
        self.logits = self.inference()

        if not is_training:
            return
        self.loss_val = self.loss()
        self.train_op = self.train()
        self.predictions = tf.argmax(self.logits, axis=1, name="predictions")

        pred = tf.nn.sigmoid(self.logits)
        label_true = tf.count_nonzero(self.input_y)
        logit_temp = tf.where(pred >= self.correct_threshold, x=pred, y=tf.zeros(tf.shape(pred)))
        logit_true = tf.count_nonzero(logit_temp)
        pred_correct = tf.count_nonzero(tf.multiply(tf.cast(self.input_y, tf.float32), logit_temp))

        self.precision = tf.divide(tf.cast(pred_correct, tf.float32), tf.cast(logit_true, tf.float32))
        self.recall = tf.divide(tf.cast(pred_correct, tf.float32), tf.cast(label_true, tf.float32))


    def instantiate_weights(self):
        with tf.name_scope("weights"):
            self.Embedding = tf.get_variable("Embedding", shape=[self.vocab_size, self.embed_size], initializer=self.initializer)
            self.left_side_first_word = tf.get_variable("left_side_first_word", shape=[self.batch_size, self.embed_size], initializer=self.initializer)
            self.right_side_last_word = tf.get_variable("right_side_last_word",shape=[self.batch_size, self.embed_size], initializer=self.initializer)
            self.W_l=tf.get_variable("W_l", shape=[self.embed_size, self.embed_size], initializer=self.initializer)
            self.W_r=tf.get_variable("W_r", shape=[self.embed_size, self.embed_size], initializer=self.initializer)
            self.W_sl=tf.get_variable("W_sl", shape=[self.embed_size, self.embed_size], initializer=self.initializer)
            self.W_sr=tf.get_variable("W_sr", shape=[self.embed_size, self.embed_size], initializer=self.initializer)

            self.W_projection = tf.get_variable("W_projection",shape=[self.hidden_size * 3, self.num_classes], initializer=self.initializer)
            self.b_projection = tf.get_variable("b_projection",shape=[self.num_classes])

    def get_context_left(self, context_left, embedding_previous):
        left_c = tf.matmul(context_left, self.W_l)
        left_e = tf.matmul(embedding_previous, self.W_sl)
        left_h = left_c+left_e
        context_left = self.activation(left_h)
        return context_left

    def get_context_right(self, context_right, embedding_afterward):
        right_c = tf.matmul(context_right, self.W_r)
        right_e = tf.matmul(embedding_afterward, self.W_sr)
        right_h = right_c + right_e
        context_right = self.activation(right_h)
        return context_right

    def conv_layer_with_recurrent_structure(self):
        embedded_words_split = tf.split(self.embedded_words, self.sequence_length, axis=1)  # [seq, None, 1, emb]
        embedded_words_squeezed = [tf.squeeze(x, axis=1) for x in embedded_words_split]  # [seq, None, emb]
        embedding_previous = self.left_side_first_word
        context_left_previous = tf.zeros((self.batch_size, self.embed_size))
        context_left_list = []
        for i, current_embedding_word in enumerate(embedded_words_squeezed):
            context_left = self.get_context_left(context_left_previous, embedding_previous)
            context_left_list.append(context_left)
            embedding_previous = current_embedding_word
            context_left_previous = context_left
        embedded_words_squeezed2 = copy.copy(embedded_words_squeezed)
        embedded_words_squeezed2.reverse()
        embedding_afterward = self.right_side_last_word
        context_right_afterward = tf.zeros((self.batch_size, self.embed_size))
        context_right_list = []
        for j, current_embedding_word in enumerate(embedded_words_squeezed2):
            context_right = self.get_context_right(context_right_afterward, embedding_afterward)
            context_right_list.append(context_right)
            embedding_afterward = current_embedding_word
            context_right_afterward = context_right
        output_list = []
        for index, current_embedding_word in enumerate(embedded_words_squeezed):
            representation = tf.concat([context_left_list[index], current_embedding_word, context_right_list[index]], axis=1)
            output_list.append(representation)  # [seq, None, emb*3]
        output = tf.stack(output_list, axis=1)  # [None, seq, emb*3]
        return output

    def inference(self):
        self.embedded_words = tf.nn.embedding_lookup(self.Embedding, self.input_x)  # [None, seq, emb]
        output_conv = self.conv_layer_with_recurrent_structure()  # [None, seq, emb*3]
        output_pooling=tf.reduce_max(output_conv, axis=1)  # [None, emb*3]
        with tf.name_scope("dropout"):
            h_drop = tf.nn.dropout(output_pooling, keep_prob=self.dropout_keep_prob) # [None, emb*3]
        with tf.name_scope("output"):
            logits = tf.matmul(h_drop, self.W_projection) + self.b_projection  # [None, num_classes]
        return logits

    def loss(self, l2_lambda=0.00001):
        with tf.name_scope("loss"):
            losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_y, logits=self.logits)
            losses = tf.reduce_sum(losses, axis=1)
            loss = tf.reduce_mean(losses)
            l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            loss = loss + l2_losses
        return loss

    def train(self):
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps, self.decay_rate, staircase=True)
        train_op = tf.contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step, learning_rate=learning_rate, optimizer="Adam")
        return train_op
