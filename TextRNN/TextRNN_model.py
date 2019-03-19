import tensorflow as tf
from tensorflow.contrib import rnn
import gensim

class TextRNN:
    def __init__(self, num_classes, learning_rate, batch_size, decay_steps, decay_rate, sequence_length,
                 embed_size, is_training, correct_threshold, word2vec_model_path, initializer=tf.random_normal_initializer(stddev=0.1)):
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.embed_size = embed_size
        self.hidden_size = embed_size
        self.is_training = is_training
        self.learning_rate = learning_rate
        self.initializer = initializer
        self.w2vModel = gensim.models.Word2Vec.load(word2vec_model_path)
        self.correct_threshold = correct_threshold

        self.input_x = tf.placeholder(tf.float32, [None, self.sequence_length, self.embed_size], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, self.num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))
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
            self.W_projection = tf.get_variable("W_projection", shape=[self.hidden_size * 2, self.num_classes],
                                                initializer=self.initializer)
            self.b_projection = tf.get_variable("b_projection", shape=[self.num_classes])

    def inference(self):
        """main computation graph here: 1. embedding layer, 2.Bi-LSTM layer, 3.concat, 4.FC layer 5.softmax """
        self.embedded_words = self.input_x
        lstm_fw_cell = rnn.BasicLSTMCell(self.hidden_size)
        lstm_bw_cell = rnn.BasicLSTMCell(self.hidden_size)
        if self.dropout_keep_prob is not None:
            lstm_fw_cell = rnn.DropoutWrapper(lstm_fw_cell,output_keep_prob=self.dropout_keep_prob)
            lstm_bw_cell = rnn.DropoutWrapper(lstm_bw_cell,output_keep_prob=self.dropout_keep_prob)
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,lstm_bw_cell,self.embedded_words,dtype=tf.float32)
        output_rnn = tf.concat(outputs,axis=2) #[batch_size,sequence_length,hidden_size*2]
        output_rnn_last = output_rnn[:,-1,:] ##[batch_size,hidden_size*2]
        with tf.name_scope("output"):
            logits = tf.matmul(output_rnn_last, self.W_projection) + self.b_projection  # [batch_size,num_classes]
        return logits

    def loss(self, l2_lambda=0.00001):
        with tf.name_scope("loss"):
            losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(self.input_y, tf.float32), logits=self.logits)
            losses = tf.reduce_sum(losses, axis=1)
            loss = tf.reduce_mean(losses)
            l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            loss = loss + l2_losses
        return loss

    def train(self):
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,self.decay_rate, staircase=True)
        train_op = tf.contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step,learning_rate=learning_rate, optimizer="Adam")
        return train_op
