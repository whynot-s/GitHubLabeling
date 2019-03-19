import sys
sys.path.append("../")
import os
import tensorflow as tf
from data_util import next_batch
from TextRNN.TextRNN_model import TextRNN
import pandas as pd

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("name_scope","rnn","name scope value.")
tf.app.flags.DEFINE_integer("num_classes", 1954, "number of label")
tf.app.flags.DEFINE_float("learning_rate", 0.01, "learning rate")
tf.app.flags.DEFINE_integer("batch_size", 50, "Batch size for training")
tf.app.flags.DEFINE_integer("decay_steps", 1000, "how many steps before decay learning rate.")
tf.app.flags.DEFINE_float("decay_rate", 0.9, "Rate of decay for learning rate.")
tf.app.flags.DEFINE_float("correct_threshold", 0.5, "threshold of correct prediction.")
tf.app.flags.DEFINE_string("ckpt_dir", "G:/GitHubLabeling/TextRNN_checkpoint/", "checkpoint location for the model")
tf.app.flags.DEFINE_integer("sequence_length", 200, "max sentence length")
tf.app.flags.DEFINE_integer("embed_size", 200, "embedding size")
tf.app.flags.DEFINE_boolean("is_training", True, "is training.true:training,false:testing/inference")
tf.app.flags.DEFINE_integer("num_epochs", 60, "number of epochs to run.")
tf.app.flags.DEFINE_integer("validate_every", 1, "Validate every validate_every epochs.")
tf.app.flags.DEFINE_string("word2vec_model_path", "G:/GraPaper/w2v/wiki.model", "word2vec's vocabulary and vectors")
tf.app.flags.DEFINE_string("train_csv_data_path", "G:/GitHubLabeling/data/readme_cleaned_filtered_1954_train.csv", "Data to train")
tf.app.flags.DEFINE_string("test_csv_data_path", "G:/GitHubLabeling/data/readme_cleaned_filtered_1954_test.csv", "Data to test")


def main(_):
    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        textRNN = TextRNN(FLAGS.num_classes, FLAGS.learning_rate, FLAGS.batch_size, FLAGS.decay_steps, FLAGS.decay_rate,
                          FLAGS.sequence_length, FLAGS.embed_size, FLAGS.is_training, FLAGS.correct_threshold,
                          FLAGS.word2vec_model_path)
        saver = tf.train.Saver()
        if os.path.exists(FLAGS.ckpt_dir + "checkpoint"):
            print("Restoring Variables from Checkpoint")
            saver.restore(sess, tf.train.latest_checkpoint(FLAGS.ckpt_dir))
        else:
            print('Initializing Variables')
            sess.run(tf.global_variables_initializer())
        curr_epoch = sess.run(textRNN.epoch_step)
        batch_size = FLAGS.batch_size
        merged_summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter('TextRNN_with_summaries', sess.graph)
        input_data = pd.read_csv(FLAGS.train_csv_data_path)
        i = 0
        for epoch in range(curr_epoch, FLAGS.num_epochs):
            loss, prec, reca, counter = 0.0, 0.0, 0.0, 0
            while True:
                start, end = counter * batch_size, (counter + 1) * batch_size
                x, y = next_batch(input_data[start:end], batch_size, FLAGS.num_classes, FLAGS.sequence_length,
                                  textRNN.w2vModel, FLAGS.embed_size)
                if x is None:
                    break
                feed_dict = {textRNN.input_x: x,
                             textRNN.input_y: y,
                             textRNN.dropout_keep_prob: 0.5}
                curr_loss, curr_prec, curr_reca, _ = sess.run(
                    [textRNN.loss_val, textRNN.precision, textRNN.recall, textRNN.train_op], feed_dict)
                loss, counter, prec, reca = loss + curr_loss, counter + 1, prec + curr_prec, reca + curr_reca
                summmary_str = sess.run(merged_summary_op, feed_dict)
                summary_writer.add_summary(summmary_str, i)
                i += 1
                if counter % 10 == 0:
                    print("Epoch %d\tBatch %d\tTrain Loss:%.3f\tTrain Precision:%.3f%%\tTrain Recall:%.3f%%" %
                          (epoch, counter, loss / float(counter), 100 * prec / float(counter),
                           100 * reca / float(counter)))
            print("going to increment epoch counter....")
            sess.run(textRNN.epoch_increment)
            print(epoch, FLAGS.validate_every, (epoch % FLAGS.validate_every == 0))
            if epoch % FLAGS.validate_every == 0:
                eval_loss, eval_prec, eval_reca = do_eval(sess, textRNN, batch_size)
                print("Epoch %d Validation Loss:%.3f\tValidation Precision: %.3f\tValidation Recall: %.3f" %
                      (epoch, eval_loss, eval_prec, eval_reca))
                # save model to checkpoint
                save_path = FLAGS.ckpt_dir + "model.ckpt"
                saver.save(sess, save_path, global_step=epoch)


def do_eval(sess, textRNN, batch_size):
    eval_loss, eval_prec, eval_reca, eval_counter = 0.0, 0.0, 0.0, 0
    test_input_data = pd.read_csv(FLAGS.test_csv_data_path)
    while True:
        start, end = eval_counter * batch_size, (eval_counter + 1) * batch_size
        x, y = next_batch(test_input_data[start:end], batch_size, FLAGS.num_classes, FLAGS.sequence_length, textRNN.w2vModel, FLAGS.embed_size)
        if x is None:
            break
        feed_dict = {textRNN.input_x: x,
                     textRNN.input_y: y,
                     textRNN.dropout_keep_prob: 1}
        curr_eval_loss, logits, curr_eval_prec, curr_eval_reca = sess.run(
            [textRNN.loss_val, textRNN.logits, textRNN.precision, textRNN.recall], feed_dict)
        eval_loss, eval_prec, eval_reca, eval_counter = eval_loss + curr_eval_loss, eval_prec + curr_eval_prec, eval_reca + curr_eval_reca, eval_counter + 1
    return eval_loss / float(eval_counter), eval_prec / float(eval_counter), eval_reca / float(eval_counter)


if __name__ == "__main__":
    tf.app.run()
