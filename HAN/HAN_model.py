import tensorflow as tf
import gensim

class HierarchicalAttention:
    def __init__(self, num_classes, learning_rate, batch_size, decay_steps, decay_rate, sequence_length, num_sentences,
                 embed_size, hidden_size, is_training, word2vec_model_path, correct_threshold,
                 initializer=tf.random_normal_initializer(stddev=0.1),clip_gradients=5.0):

        self.num_classes = num_classes
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.num_sentences = num_sentences
        self.embed_size = embed_size
        self.learning_rate = tf.Variable(learning_rate, trainable=False, name="learning_rate")
        self.learning_rate_decay_half_op = tf.assign(self.learning_rate, self.learning_rate * 0.5)
        self.initializer = initializer
        self.hidden_size = hidden_size
        self.clip_gradients = clip_gradients
        self.correct_threshold = correct_threshold
        self.is_training = is_training
        self.w2vModel = gensim.models.Word2Vec.load(word2vec_model_path)

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
        with tf.name_scope("embedding_projection"):
            self.W_projection = tf.get_variable("W_projection", shape=[self.hidden_size * 4, self.num_classes], initializer=self.initializer)
            self.b_projection = tf.get_variable("b_projection", shape=[self.num_classes])

        with tf.name_scope("gru_weights_word_level"):
            # GRU parameters:update gate related
            self.W_z = tf.get_variable("W_z", shape=[self.embed_size, self.hidden_size], initializer=self.initializer)
            self.U_z = tf.get_variable("U_z", shape=[self.embed_size, self.hidden_size], initializer=self.initializer)
            self.b_z = tf.get_variable("b_z", shape=[self.hidden_size])
            # GRU parameters:reset gate related
            self.W_r = tf.get_variable("W_r", shape=[self.embed_size, self.hidden_size], initializer=self.initializer)
            self.U_r = tf.get_variable("U_r", shape=[self.embed_size, self.hidden_size], initializer=self.initializer)
            self.b_r = tf.get_variable("b_r", shape=[self.hidden_size])
            # GRU parameters current state related
            self.W_h = tf.get_variable("W_h", shape=[self.embed_size, self.hidden_size], initializer=self.initializer)
            self.U_h = tf.get_variable("U_h", shape=[self.embed_size, self.hidden_size], initializer=self.initializer)
            self.b_h = tf.get_variable("b_h", shape=[self.hidden_size])

        with tf.name_scope("gru_weights_sentence_level"):
            # GRU parameters:update gate related
            self.W_z_sentence = tf.get_variable("W_z_sentence", shape=[self.hidden_size * 2, self.hidden_size * 2], initializer=self.initializer)
            self.U_z_sentence = tf.get_variable("U_z_sentence", shape=[self.hidden_size * 2, self.hidden_size * 2], initializer=self.initializer)
            self.b_z_sentence = tf.get_variable("b_z_sentence", shape=[self.hidden_size * 2])
            # GRU parameters:reset gate related
            self.W_r_sentence = tf.get_variable("W_r_sentence", shape=[self.hidden_size * 2, self.hidden_size * 2], initializer=self.initializer)
            self.U_r_sentence = tf.get_variable("U_r_sentence", shape=[self.hidden_size * 2, self.hidden_size * 2], initializer=self.initializer)
            self.b_r_sentence = tf.get_variable("b_r_sentence", shape=[self.hidden_size * 2])
            # GRU parameters current state related
            self.W_h_sentence = tf.get_variable("W_h_sentence", shape=[self.hidden_size * 2, self.hidden_size * 2], initializer=self.initializer)
            self.U_h_sentence = tf.get_variable("U_h_sentence", shape=[self.hidden_size * 2, self.hidden_size * 2], initializer=self.initializer)
            self.b_h_sentence = tf.get_variable("b_h_sentence", shape=[self.hidden_size * 2])

        with tf.name_scope("attention"):
            self.W_w_attention_word = tf.get_variable("W_w_attention_word", shape=[self.hidden_size * 2, self.hidden_size * 2], initializer=self.initializer)
            self.W_b_attention_word = tf.get_variable("W_b_attention_word", shape=[self.hidden_size * 2])
            self.W_w_attention_sentence = tf.get_variable("W_w_attention_sentence", shape=[self.hidden_size * 4, self.hidden_size * 2], initializer=self.initializer)
            self.W_b_attention_sentence = tf.get_variable("W_b_attention_sentence", shape=[self.hidden_size * 2])
            self.context_vecotor_word = tf.get_variable("what_is_the_informative_word", shape=[self.hidden_size * 2], initializer=self.initializer)
            self.context_vecotor_sentence = tf.get_variable("what_is_the_informative_sentence", shape=[self.hidden_size * 2], initializer=self.initializer)

    def inference(self):
        self.embedded_words = tf.split(self.input_x, self.num_sentences, axis=1)
        self.embedded_words = tf.stack(self.embedded_words, axis=1)
        embedded_words_reshaped = tf.reshape(self.embedded_words, shape=[-1, int(self.sequence_length / self.num_sentences), self.embed_size])
        # shape: [batch_size*num_sentences,sentence_length,embed_size]
        hidden_state_forward_list = self.gru_forward_word_level(embedded_words_reshaped)
        hidden_state_backward_list = self.gru_backward_word_level(embedded_words_reshaped)
        # shape: [sentence_length, batch_size*num_sentences,hidden_size]
        self.hidden_state = [tf.concat([h_forward, h_backward], axis=1) for h_forward, h_backward in zip(hidden_state_forward_list, hidden_state_backward_list)]
        # shape: [sentence_length, batch_size*num_sentences,hidden_size*2]

        sentence_representation = self.attention_word_level(self.hidden_state)
        # shape: [batch_size*num_sentences,hidden_size*2]
        sentence_representation = tf.reshape(sentence_representation, shape=[-1, self.num_sentences, self.hidden_size * 2])
        # shape: [batch_size,num_sentences,hidden_size*2]

        hidden_state_forward_sentences = self.gru_forward_sentence_level(sentence_representation)
        hidden_state_backward_sentences = self.gru_backward_sentence_level(sentence_representation)
        self.hidden_state_sentence = [tf.concat([h_forward, h_backward], axis=1) for h_forward, h_backward in zip(hidden_state_forward_sentences, hidden_state_backward_sentences)]
        document_representation = self.attention_sentence_level(self.hidden_state_sentence)
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(document_representation, keep_prob=self.dropout_keep_prob)
        with tf.name_scope("output"):
            logits = tf.matmul(self.h_drop, self.W_projection) + self.b_projection
        return logits

    def gru_single_step_word_level(self, Xt, h_t_minus_1):
        # update gate
        z_t = tf.nn.sigmoid(tf.matmul(Xt, self.W_z) + tf.matmul(h_t_minus_1, self.U_z) + self.b_z)
        # reset gate
        r_t = tf.nn.sigmoid(tf.matmul(Xt, self.W_r) + tf.matmul(h_t_minus_1, self.U_r) + self.b_r)
        # candiate state h_t~
        h_t_candiate = tf.nn.tanh(tf.matmul(Xt, self.W_h) + r_t * (tf.matmul(h_t_minus_1, self.U_h)) + self.b_h)
        # new state
        h_t = (1 - z_t) * h_t_minus_1 + z_t * h_t_candiate
        return h_t

    def gru_forward_word_level(self, embedded_words):
        embedded_words_splitted = tf.split(embedded_words, int(self.sequence_length / self.num_sentences), axis=1)
        embedded_words_squeeze = [tf.squeeze(x, axis=1) for x in embedded_words_splitted]
        h_t = tf.ones((self.batch_size * self.num_sentences, self.hidden_size))
        h_t_forward_list = []
        for time_step, Xt in enumerate(embedded_words_squeeze):
            h_t = self.gru_single_step_word_level(Xt, h_t)
            h_t_forward_list.append(h_t)
        # shape: [sentence_length, batch_size*num_sentences, hidden_size]
        return h_t_forward_list

    # backward gru for first level: word level
    def gru_backward_word_level(self, embedded_words):
        embedded_words_splitted = tf.split(embedded_words, int(self.sequence_length / self.num_sentences), axis=1)
        embedded_words_squeeze = [tf.squeeze(x, axis=1) for x in embedded_words_splitted]
        embedded_words_squeeze.reverse()
        h_t = tf.ones((self.batch_size * self.num_sentences, self.hidden_size))
        h_t_backward_list = []
        for time_step, Xt in enumerate(embedded_words_squeeze):
            h_t = self.gru_single_step_word_level(Xt, h_t)
            h_t_backward_list.append(h_t)
        # shape: [sentence_length, batch_size*num_sentences, hidden_size]
        h_t_backward_list.reverse()
        return h_t_backward_list

    def attention_word_level(self, hidden_state):
        hidden_state_ = tf.stack(hidden_state, axis=1)
        # shape: [batch_size*num_sentences, sentence_length, hidden_size*2]
        hidden_state_2 = tf.reshape(hidden_state_, shape=[-1, self.hidden_size * 2])
        # shape: [batch_size*num_sentences*sentence_length, hidden_size*2]
        hidden_representation = tf.nn.tanh(tf.matmul(hidden_state_2, self.W_w_attention_word) + self.W_b_attention_word)
        hidden_representation = tf.reshape(hidden_representation, shape=[-1, int(self.sequence_length / self.num_sentences), self.hidden_size * 2])

        hidden_state_context_similiarity = tf.multiply(hidden_representation, self.context_vecotor_word)
        # shape: [batch_size*num_sentences,sentence_length,hidden_size*2]
        attention_logits = tf.reduce_sum(hidden_state_context_similiarity, axis=2)
        # shape: [batch_size*num_sentences,sentence_length]
        attention_logits_max = tf.reduce_max(attention_logits, axis=1, keep_dims=True)
        # shape: [batch_size*num_sentences,1]
        p_attention = tf.nn.softmax(attention_logits - attention_logits_max)
        # shape:[batch_size*num_sentences,sentence_length_length]
        p_attention_expanded = tf.expand_dims(p_attention, axis=2)
        sentence_representation = tf.multiply(p_attention_expanded, hidden_state_)
        sentence_representation = tf.reduce_sum(sentence_representation, axis=1)
        return sentence_representation  # shape:[batch_size*num_sentences,hidden_size*2]

    def gru_forward_sentence_level(self, sentence_representation):
        sentence_representation_splitted = tf.split(sentence_representation, self.num_sentences, axis=1)
        sentence_representation_squeeze = [tf.squeeze(x, axis=1) for x in sentence_representation_splitted]
        h_t = tf.ones((self.batch_size, self.hidden_size * 2))
        h_t_forward_list = []
        for time_step, Xt in enumerate(sentence_representation_squeeze):
            h_t = self.gru_single_step_sentence_level(Xt, h_t)
            h_t_forward_list.append(h_t)
        return h_t_forward_list

    def gru_backward_sentence_level(self, sentence_representation):
        sentence_representation_splitted = tf.split(sentence_representation, self.num_sentences, axis=1)
        sentence_representation_squeeze = [tf.squeeze(x, axis=1) for x in sentence_representation_splitted]
        sentence_representation_squeeze.reverse()
        h_t = tf.ones((self.batch_size, self.hidden_size * 2))
        h_t_forward_list = []
        for time_step, Xt in enumerate(sentence_representation_squeeze):
            h_t = self.gru_single_step_sentence_level(Xt,h_t)
            h_t_forward_list.append(h_t)
        h_t_forward_list.reverse()
        return h_t_forward_list

    def gru_single_step_sentence_level(self, Xt, h_t_minus_1):
        z_t = tf.nn.sigmoid(tf.matmul(Xt, self.W_z_sentence) + tf.matmul(h_t_minus_1, self.U_z_sentence) + self.b_z_sentence)
        r_t = tf.nn.sigmoid(tf.matmul(Xt, self.W_r_sentence) + tf.matmul(h_t_minus_1, self.U_r_sentence) + self.b_r_sentence)
        h_t_candidate = tf.nn.tanh(tf.matmul(Xt, self.W_h_sentence) + r_t * (tf.matmul(h_t_minus_1, self.U_h_sentence)) + self.b_h_sentence)
        h_t = (1 - z_t) * h_t_minus_1 + z_t * h_t_candidate
        return h_t

    def attention_sentence_level(self, hidden_state_sentence):
        hidden_state_ = tf.stack(hidden_state_sentence, axis=1)
        hidden_state_2 = tf.reshape(hidden_state_, shape=[-1, self.hidden_size * 4])
        hidden_representation = tf.nn.tanh(tf.matmul(hidden_state_2, self.W_w_attention_sentence) + self.W_b_attention_sentence)
        hidden_representation = tf.reshape(hidden_representation, shape=[-1, self.num_sentences, self.hidden_size * 2])
        hidden_state_context_similiarity = tf.multiply(hidden_representation, self.context_vecotor_sentence)
        attention_logits = tf.reduce_sum(hidden_state_context_similiarity, axis=2)
        attention_logits_max = tf.reduce_max(attention_logits, axis=1, keep_dims=True)
        p_attention = tf.nn.softmax(attention_logits - attention_logits_max)
        p_attention_expanded = tf.expand_dims(p_attention, axis=2)
        sentence_representation = tf.multiply(p_attention_expanded, hidden_state_)
        sentence_representation = tf.reduce_sum(sentence_representation, axis=1)
        return sentence_representation

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
        self.learning_rate_ = learning_rate
        train_op = tf.contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step, learning_rate=learning_rate, optimizer="Adam",clip_gradients=self.clip_gradients)
        return train_op

