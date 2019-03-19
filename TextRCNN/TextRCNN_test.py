import sys
sys.path.append("../")
import os
import tensorflow as tf
from data_util import next_batch
from TextRCNN.TextRCNN_model import TextRCNN
import multiprocessing
import pandas as pd


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("num_classes", 1954, "number of label")
tf.app.flags.DEFINE_float("learning_rate", 0.01, "learning rate")
tf.app.flags.DEFINE_integer("batch_size", 50, "Batch size for training")
tf.app.flags.DEFINE_integer("decay_steps", 6000, "how many steps before decay learning rate.")
tf.app.flags.DEFINE_float("decay_rate", 0.9, "Rate of decay for learning rate.")
tf.app.flags.DEFINE_float("correct_threshold", 0.5, "threshold of correct prediction.")
tf.app.flags.DEFINE_string("ckpt_dir", "G:/GitHubLabeling/TextRCNN_checkpoint/", "checkpoint location for the model")
tf.app.flags.DEFINE_integer("sequence_length", 200, "max sentence length")
tf.app.flags.DEFINE_integer("embed_size", 200, "embedding size")
tf.app.flags.DEFINE_boolean("is_training", True, "is training.true:training,false:testing/inference")
tf.app.flags.DEFINE_integer("num_epochs", 60, "number of epochs to run.")
tf.app.flags.DEFINE_integer("validate_every", 1, "Validate every validate_every epochs.")
tf.app.flags.DEFINE_string("word2vec_model_path", "G:/GraPaper/w2v/wiki.model", "word2vec's vocabulary and vectors")
tf.app.flags.DEFINE_string("train_csv_data_path", "G:/GitHubLabeling/data/readme_cleaned_filtered_1954_train.csv", "Data to train")
tf.app.flags.DEFINE_string("test_csv_data_path", "G:/GitHubLabeling/data/readme_cleaned_filtered_1954_test.csv", "Data to test")


def main(_):
    vocab_size = 0
    config = tf.ConfigProto(
        device_count={"CPU": multiprocessing.cpu_count()},
        inter_op_parallelism_threads=0,
        intra_op_parallelism_threads=0,
        log_device_placement=True
    )
    with tf.Session(config=config) as sess:
        textRCNN = TextRCNN(FLAGS.num_classes, FLAGS.learning_rate, FLAGS.batch_size, FLAGS.decay_steps,
                            FLAGS.decay_rate, FLAGS.sequence_length, vocab_size, FLAGS.embed_size, FLAGS.is_training,
                            FLAGS.correct_threshold, FLAGS.word2vec_model_path)
        saver = tf.train.Saver()
        if os.path.exists(FLAGS.ckpt_dir+"checkpoint"):
            print("Restoring Variables from Checkpoint")
            saver.restore(sess, tf.train.latest_checkpoint(FLAGS.ckpt_dir))
        else:
            print("No model found at %s" % FLAGS.ckpt_dir + "checkpoint")
            return
        eval_loss, eval_prec, eval_reca = do_eval(sess, textRCNN, FLAGS.batch_size, FLAGS.num_classes, FLAGS.correct_threshold)
        print("Validation Loss:%.3f\tValidation Precision: %.3f\tValidation Recall: %.3f" %
              (eval_loss, eval_prec, eval_reca))


def do_eval(sess, textRCNN, batch_size, num_classes, correct_threshold):
    eval_loss, eval_prec, eval_reca, eval_counter = 0.0, 0.0, 0.0, 0
    test_input_data = pd.read_csv(FLAGS.test_csv_data_path)
    while True:
        start, end = eval_counter * batch_size, (eval_counter + 1) * batch_size
        x, y = next_batch(test_input_data[start:end], batch_size, FLAGS.num_classes, FLAGS.sequence_length, textRCNN.w2vModel, FLAGS.embed_size)
        if x is None:
            break
        feed_dict = {textRCNN.input_x: x,
                     textRCNN.input_y: y,
                     textRCNN.dropout_keep_prob: 1}
        curr_eval_loss, prediction = sess.run(
            [textRCNN.loss_val, textRCNN.prediction], feed_dict)
        eval_loss += curr_eval_loss
        eval_counter += 1
        for i in range(batch_size):
            label_true = 0.0
            pred_true = 0.0
            pred_correct = 0.0
            for j in range(num_classes):
                if y[i][j] == 1:
                    label_true += 1.0
                if prediction[i][j] >= correct_threshold:
                    pred_true += 1.0
                if y[i][j] == 1 and prediction[i][j] >= correct_threshold:
                    pred_correct += 1.0
            if pred_true > 0:
                eval_prec += (pred_correct / pred_true)
            eval_reca += (pred_correct / label_true)
    return eval_loss / float(eval_counter*batch_size), eval_prec / float(eval_counter*batch_size), eval_reca / float(eval_counter*batch_size)


if __name__ == "__main__":
    tf.app.run()