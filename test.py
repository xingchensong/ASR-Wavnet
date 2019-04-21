import tensorflow as tf
import argparse
import os,math,copy,shutil
from utils.mylogger import *
from utils import ValueWindow , plot
from hparams import hparams as hp
from hparams import hparams_debug_string
from datafeeder import DataFeeder,DataFeeder_wavnet
from models import create_model
import datetime,time
import traceback,random
import numpy as np
from keras import backend as K


def add_stats(model):
  with tf.variable_scope('test_stats') as scope:
    tf.summary.histogram('pred_labels', model.decoded2)
    tf.summary.histogram('labels', model.labels)
    tf.summary.scalar('batch_loss', model.batch_loss)
    tf.summary.scalar('batch_wer', model.WER)
    return tf.summary.merge_all()

def test():
    parser = argparse.ArgumentParser()
    # TODO: add arguments
    parser.add_argument('--log_dir', default=os.path.expanduser('~/my_asr2/logdir/logging'))
    parser.add_argument('--serving_dir', default=os.path.expanduser('~/my_asr2/logdir/serving_am/'))
    parser.add_argument('--data_dir', default=os.path.expanduser('~/corpus_zn'))
    parser.add_argument('--model', default='ASR_wavnet')
    # parser.add_argument('--epochs', type=int, help='Max epochs to run.', default=100)
    parser.add_argument('--restore_step', type=int, help='Global step to restore from checkpoint.', default=2100)
    parser.add_argument('--serving', type=bool, help='', default=False)
    # parser.add_argument('--validation_interval', type=int, help='一个epoch验证5次，每次200步共3200条数据', default=7090) # 35450//5
    parser.add_argument('--summary_interval', type=int, default=1, help='Steps between running summary ops.')
    # parser.add_argument('--checkpoint_interval', type=int, default=100, help='Steps between writing checkpoints.')
    parser.add_argument('--hparams', default='',
                        help='Hyperparameter overrides as a comma-separated list of name=value pairs')
    args = parser.parse_args()

    run_name = args.model
    logdir = os.path.join(args.log_dir, 'logs-%s' % run_name)
    init(os.path.join(logdir, 'test.log'), run_name)
    hp.parse(args.hparams)

    # TODO：parse  ckpt,arguments,hparams
    checkpoint_path = os.path.join(logdir, 'model.ckpt')
    input_path = args.data_dir
    log('Checkpoint path: %s' % checkpoint_path)
    log('Loading training data from : %s ' % input_path)
    log('Using model : %s' % args.model)

    # TODO：set up datafeeder
    with tf.variable_scope('datafeeder') as scope:
        hp.data_type = 'test'
        hp.feature_type = 'mfcc'
        hp.data_length = None
        hp.initial_learning_rate = 0.0005
        hp.batch_size = 1
        hp.aishell = False
        hp.prime = False
        hp.stcmd = False
        hp.AM = True
        hp.LM = False
        hp.shuffle = False
        hp.is_training = False # TODO: 在infer的时候一定要设置为False否则bn会扰乱所有的值！
        feeder = DataFeeder_wavnet(args=hp)
        log('num_wavs:' + str(len(feeder.wav_lst)))

        feeder.am_vocab = np.load('logdir/am_pinyin_dict.npy').tolist()
        hp.input_vocab_size = len(feeder.am_vocab)
        hp.final_output_dim = len(feeder.am_vocab)
        hp.steps_per_epoch = len(feeder.wav_lst) // hp.batch_size
        log('steps_per_epoch:' + str(hp.steps_per_epoch))
        log('pinyin_vocab_size:' + str(hp.input_vocab_size))

    # TODO: set up model
    with tf.variable_scope('model') as scope:
        model = create_model(args.model, hp)
        model.build_graph()
        model.add_loss()
        model.add_decoder()
        # model.add_optimizer(global_step=global_step)
        # TODO: summary
        stats = add_stats(model)

    # TODO：Set up saver and Bookkeeping
    time_window = ValueWindow(100)
    loss_window = ValueWindow(100)
    wer_window = ValueWindow(100)
    saver = tf.train.Saver(max_to_keep=20)

    # TODO: test
    with tf.Session(graph=tf.get_default_graph()) as sess:

        log(hparams_debug_string(hp))
        try:
            # TODO: Set writer and initializer
            summary_writer = tf.summary.FileWriter(logdir + '/test', sess.graph)
            sess.run(tf.global_variables_initializer())

            # TODO: Restore
            if args.restore_step:
                # Restore from a checkpoint if the user requested it.
                restore_path = '%s-%d' % (checkpoint_path, args.restore_step)
                saver.restore(sess, restore_path)
                log('Resuming from checkpoint: %s ' % restore_path)
            else:
                log('Starting new training run ')

            # TODO: epochs steps batch
            step = 0
            batch_data = feeder.get_am_batch()
            for j in range(hp.steps_per_epoch):
                input_batch = next(batch_data)
                feed_dict = {model.inputs: input_batch['the_inputs'],
                             model.labels: input_batch['the_labels'],
                             model.input_lengths: input_batch['input_length'],
                             model.label_lengths: input_batch['label_length']}
                # TODO: Run one step
                start_time = time.time()
                array_loss, batch_loss,wer,label,final_pred_label = sess.run([ model.ctc_loss,model.batch_loss,model.WER,model.labels, model.decoded1],
                                                      feed_dict=feed_dict)
                time_window.append(time.time() - start_time)
                step = step+1

                # TODO: Append loss
                loss_window.append(batch_loss)
                wer_window.append(wer)
                message = 'Step %-7d [%.03f sec/step, loss=%.05f, avg_loss=%.05f, wer=%.05f, avg_wer=%.05f]' % (
                    step, time_window.average, batch_loss, loss_window.average, wer,wer_window.average)
                log(message)
                # TODO: show pred and write summary
                log('label.shape           :' + str(label.shape))  # (batch_size , label_length)
                log('final_pred_label.shape:' + str(
                    np.asarray(final_pred_label).shape))
                log('label           : ' + str(label[0]))
                log('final_pred_label: ' + str(np.asarray(final_pred_label)[0][0]))

                log('Writing summary at step: %d' % step)
                summary_writer.add_summary(sess.run(stats, feed_dict=feed_dict), step)

                # TODO: Check loss
                if math.isnan(batch_loss):
                    log('Loss exploded to %.05f at step %d!' % (batch_loss, step))
                    raise Exception('Loss Exploded')

            log('serving step: ' + str(step))
            # TODO: Set up serving builder and signature map
            serve_dir = args.serving_dir + '0001'
            if os.path.exists(serve_dir):
                shutil.rmtree(serve_dir)
                log('delete exists dirs:' + serve_dir)
            builder = tf.saved_model.builder.SavedModelBuilder(export_dir=serve_dir)
            input_spec = tf.saved_model.utils.build_tensor_info(model.inputs)
            input_len = tf.saved_model.utils.build_tensor_info(model.input_lengths)
            output_labels = tf.saved_model.utils.build_tensor_info(model.decoded1)
            output_logits = tf.saved_model.utils.build_tensor_info(model.pred_softmax)
            prediction_signature = (
                tf.saved_model.signature_def_utils.build_signature_def(
                    inputs={'mfcc': input_spec, 'len': input_len},
                    outputs={'label': output_labels, 'logits': output_logits},
                    method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
            )

            builder.add_meta_graph_and_variables(
                sess=sess, tags=[tf.saved_model.tag_constants.SERVING],
                signature_def_map={
                    'predict_AudioSpec2Pinyin':
                        prediction_signature,
                },
                main_op=tf.tables_initializer(),
                strip_default_attrs=False
            )
            builder.save()
            log('Done store serving-model')

        except Exception as e:
            log('Exiting due to exception: %s' % e)
            traceback.print_exc()

if __name__=='__main__':
    test()