import sys
sys.path.append("../")
import os
import gensim
import numpy as np
import tensorflow as tf
from data_util import create_vocabulary, next_batch
from TextRCNN.TextRCNN_model import TextRCNN

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("num_classes", 1954, "number of label")
tf.app.flags.DEFINE_float("learning_rate", 0.01, "learning rate")
tf.app.flags.DEFINE_integer("batch_size", 50, "Batch size for training")
tf.app.flags.DEFINE_integer("decay_steps", 6000, "how many steps before decay learning rate.")
tf.app.flags.DEFINE_float("decay_rate", 0.9, "Rate of decay for learning rate.")
tf.app.flags.DEFINE_float("correct_threshold", 0.5, "threshold of correct prediction.")
tf.app.flags.DEFINE_string("ckpt_dir", "/sdpdata2/wjrj/GitHubLabeling/TextRCNN_checkpoint/", "checkpoint location for the model")
tf.app.flags.DEFINE_integer("sequence_length", 200, "max sentence length")
tf.app.flags.DEFINE_integer("embed_size", 200, "embedding size")
tf.app.flags.DEFINE_boolean("is_training", True, "is training.true:training,false:testing/inference")
tf.app.flags.DEFINE_integer("num_epochs", 60, "number of epochs to run.")
tf.app.flags.DEFINE_integer("validate_every", 1, "Validate every validate_every epochs.")
tf.app.flags.DEFINE_string("word2vec_model_path", "/sdpdata2/wjrj/w2v/wiki.model", "word2vec's vocabulary and vectors")


def main(_):
    vocabulary_word2index, vocabulary_index2word = create_vocabulary(FLAGS.word2vec_model_path, "TextRCNN")
    vocab_size = len(vocabulary_word2index)
    print("text_rcnn_model.vocab_size:", vocab_size)
    config = tf.ConfigProto(
        device_count={"CPU": 8},
        nter_op_parallelism_threads=0,
        intra_op_parallelism_threads=0,
        log_device_placement=True
    )
    with tf.Session(config=config) as sess:
        textRCNN = TextRCNN(FLAGS.num_classes, FLAGS.learning_rate, FLAGS.batch_size, FLAGS.decay_steps,
                            FLAGS.decay_rate, FLAGS.sequence_length, vocab_size, FLAGS.embed_size, FLAGS.is_training,
                            FLAGS.batch_size, FLAGS.correct_threshold)
        saver = tf.train.Saver()
        if os.path.exists(FLAGS.ckpt_dir+"checkpoint"):
            print("Restoring Variables from Checkpoint")
            saver.restore(sess, tf.train.latest_checkpoint(FLAGS.ckpt_dir))
        else:
            print('Initializing Variables')
            sess.run(tf.global_variables_initializer())
            assign_pretrained_word_embedding(sess, vocabulary_index2word, vocab_size, textRCNN, word2vec_model_path=FLAGS.word2vec_model_path)
        curr_epoch = sess.run(textRCNN.epoch_step)
        batch_size = FLAGS.batch_size
        for epoch in range(curr_epoch, FLAGS.num_epochs):
            loss, prec, reca, counter = 0.0, 0.0, 0.0, 0
            while True:
                x, y = next_batch(counter, batch_size, FLAGS.num_classes, FLAGS.sequence_length, training=True)
                if x is None:
                    break
                feed_dict = {textRCNN.input_x: x,
                             textRCNN.input_y: y,
                             textRCNN.dropout_keep_prob: 0.5}
                curr_loss, curr_prec, curr_reca, _ = sess.run([textRCNN.loss_val, textRCNN.precision, textRCNN.recall, textRCNN.train_op], feed_dict)
                loss, counter, acc, reca = loss + curr_loss, counter + 1, prec + curr_prec, reca + curr_reca
                if counter % 50 == 0:
                    print("Epoch %d\tBatch %d\tTrain Loss:%.3f\tTrain Precision:%.3f\tTrain Recall:%.3f" %
                          (epoch, counter, loss / float(counter), prec / float(counter), reca / float(reca)))
            print("going to increment epoch counter....")
            sess.run(textRCNN.epoch_increment)
            print(epoch, FLAGS.validate_every, (epoch % FLAGS.validate_every == 0))
            if epoch % FLAGS.validate_every == 0:
                eval_loss, eval_prec, eval_reca = do_eval(sess, textRCNN, batch_size)
                print("Epoch %d Validation Loss:%.3f\tValidation Precision: %.3f\tValidation Recall: %.3f" %
                      (epoch, eval_loss, eval_prec, eval_reca))
                # save model to checkpoint
                save_path = FLAGS.ckpt_dir + "model.ckpt"
                saver.save(sess, save_path, global_step=epoch)


def assign_pretrained_word_embedding(sess, vocabulary_index2word, vocab_size, textRCNN, word2vec_model_path=None):
    print("using pre-trained word embedding.started.word2vec_model_path:", word2vec_model_path)
    word2vec_model = gensim.models.Word2Vec.load(word2vec_model_path)
    word2vec_dict = {}
    for word in word2vec_model.wv.vocab:
        word2vec_dict[word] = word2vec_model.wv[word]
    word_embedding_2dlist = [[]] * vocab_size
    word_embedding_2dlist[0] = np.zeros(FLAGS.embed_size)
    bound = np.sqrt(6.0) / np.sqrt(vocab_size)
    count_exist = 0
    count_not_exist = 0
    for i in range(1, vocab_size):
        word = vocabulary_index2word[i]
        embedding = None
        try:
            embedding = word2vec_dict[word]
        except Exception:
            embedding = None
        if embedding is not None:
            word_embedding_2dlist[i] = embedding
            count_exist = count_exist + 1
        else:
            word_embedding_2dlist[i] = np.random.uniform(-bound, bound, FLAGS.embed_size)
            count_not_exist = count_not_exist + 1
    word_embedding_final = np.array(word_embedding_2dlist)
    word_embedding = tf.constant(word_embedding_final, dtype=tf.float32)
    t_assign_embedding = tf.assign(textRCNN.Embedding, word_embedding)
    sess.run(t_assign_embedding)
    print("word. exists embedding:", count_exist, " ;word not exist embedding:", count_not_exist)
    print("using pre-trained word embedding.ended...")


def do_eval(sess, textRCNN, batch_size):
    eval_loss, eval_prec, eval_reca, eval_counter = 0.0, 0.0, 0.0, 0
    while True:
        x, y = next_batch(eval_counter, batch_size, FLAGS.num_classes, FLAGS.sequence_length, training=False)
        if x is None:
            break
        feed_dict = {textRCNN.input_x: x,
                     textRCNN.input_y: y,
                     textRCNN.dropout_keep_prob: 1}
        curr_eval_loss, logits, curr_eval_prec, curr_eval_reca = sess.run(
            [textRCNN.loss_val, textRCNN.logits, textRCNN.precision, textRCNN.recall], feed_dict)
        eval_loss, eval_prec, eval_reca, eval_counter = eval_loss + curr_eval_loss, eval_prec + curr_eval_prec, eval_reca + curr_eval_reca, eval_counter + 1
    return eval_loss / float(eval_counter), eval_prec / float(eval_counter), eval_reca / float(eval_counter)


if __name__ == "__main__":
    tf.app.run()
