import tensorflow as tf
import argparse
import os,math,copy
from utils.mylogger import *
from utils import ValueWindow , plot
from hparams import hparams as hp
from hparams import hparams_debug_string
from datafeeder import DataFeeder,DataFeeder_wavnet,DataFeeder_transformer
from models import create_model
import datetime,time
import traceback,random
import numpy as np
from keras import backend as K


def add_stats(model):
  with tf.variable_scope('stats') as scope:
    tf.summary.scalar('acc', model.acc)
    tf.summary.scalar('mean_loss', model.mean_loss)
    # tf.summary.scalar('learning_rate', model.learning_rate)
    # gradient_norms = [tf.norm(grad) for grad in model.gradients]
    # tf.summary.histogram('gradient_norm', gradient_norms)
    # tf.summary.scalar('max_gradient_norm', tf.reduce_max(gradient_norms))
    return tf.summary.merge_all()

def add_dev_stats(model):
    with tf.variable_scope('dev_stats') as scope:
        tf.summary.scalar('dev_mean_loss',model.mean_loss)
        tf.summary.scalar('dev_acc', model.acc)
        return tf.summary.merge_all()

def time_string():
  return datetime.now().strftime('%Y-%m-%d %H:%M')


def train(logdir,args):

    # TODO：parse  ckpt,arguments,hparams
    checkpoint_path = os.path.join(logdir,'model.ckpt')
    input_path = args.data_dir
    log('Checkpoint path: %s' % checkpoint_path)
    log('Loading training data from : %s ' % input_path)
    log('Using model : %s' %args.model)

    # TODO：set up datafeeder
    with tf.variable_scope('datafeeder') as scope:
        hp.data_length = None
        hp.initial_learning_rate = 0.0001
        hp.batch_size = 256
        hp.prime = True
        hp.stcmd = True
        feeder = DataFeeder(args=hp)
        log('num_sentences:'+str(len(feeder.wav_lst))) # 283600
        hp.input_vocab_size = len(feeder.pny_vocab)
        hp.final_output_dim = len(feeder.pny_vocab)
        hp.steps_per_epoch = len(feeder.wav_lst)//hp.batch_size
        log('steps_per_epoch:' + str(hp.steps_per_epoch))  # 17725
        log('pinyin_vocab_size:'+str(hp.input_vocab_size)) # 1292
        hp.label_vocab_size = len(feeder.han_vocab)
        log('label_vocab_size :' + str(hp.label_vocab_size)) # 6291

    # TODO：set up model
    global_step = tf.Variable(initial_value=0,name='global_step',trainable=False)
    valid_step = 0
    # valid_global_step = tf.Variable(initial_value=0,name='valid_global_step',trainable=False)
    with tf.variable_scope('model') as scope:
        model = create_model(args.model,hp)
        model.build_graph()
        model.add_loss()
        model.add_optimizer(global_step=global_step,loss=model.mean_loss)
        # TODO: summary
        stats = add_stats(model=model)
        valid_stats = add_dev_stats(model)


    # TODO：Set up saver and Bookkeeping
    time_window = ValueWindow(100)
    loss_window = ValueWindow(100)
    acc_window = ValueWindow(100)
    valid_time_window = ValueWindow(100)
    valid_loss_window = ValueWindow(100)
    valid_acc_window = ValueWindow(100)
    saver = tf.train.Saver(max_to_keep=20)
    first_serving = True
    # TODO: train
    with tf.Session() as sess:

        log(hparams_debug_string(hp))
        try:
            # TODO: Set writer and initializer
            summary_writer = tf.summary.FileWriter(logdir + '/train', sess.graph)
            summary_writer_dev = tf.summary.FileWriter(logdir + '/dev')
            sess.run(tf.global_variables_initializer())

            # TODO: Restore
            if args.restore_step:
                # Restore from a checkpoint if the user requested it.
                restore_path = '%s-%d' % (checkpoint_path, args.restore_step)
                saver.restore(sess, restore_path)
                log('Resuming from checkpoint: %s ' % restore_path)
            else:
                log('Starting new training run ')

            step = 0
            # TODO: epochs steps batch
            for i in range(args.epochs):
                batch_data = feeder.get_lm_batch()
                log('Traning epoch '+ str(i)+':')
                for j in range(hp.steps_per_epoch):
                    input_batch, label_batch = next(batch_data)
                    feed_dict = {
                        model.x:input_batch,
                        model.y:label_batch,
                    }
                    # TODO: Run one step ~~~
                    start_time = time.time()
                    total_step,batch_loss,batch_acc,opt = sess.run([global_step, model.mean_loss,model.acc,model.optimize],feed_dict=feed_dict)
                    time_window.append(time.time() - start_time)
                    step = total_step

                    # TODO: Append loss
                    loss_window.append(batch_loss)
                    acc_window.append(batch_acc)
                    message = 'Step %-7d [%.03f sec/step, loss=%.05f, avg_loss=%.05f,acc=%.05f, avg_acc=%.05f,  lr=%.07f]' % (
                        step, time_window.average, batch_loss, loss_window.average,batch_acc,acc_window.average,K.get_value(model.learning_rate))
                    log(message)

                    # TODO: Check loss
                    if math.isnan(batch_loss):
                        log('Loss exploded to %.05f at step %d!' % (batch_loss, step))
                        raise Exception('Loss Exploded')

                    # TODO: Check sumamry
                    if step % args.summary_interval == 0:
                        log('Writing summary at step: %d' % step)
                        summary_writer.add_summary(sess.run(stats,feed_dict=feed_dict), step)

                    # TODO: Check checkpoint
                    if step % args.checkpoint_interval == 0:
                        log('Saving checkpoint to: %s-%d' % (checkpoint_path, step))
                        saver.save(sess, checkpoint_path, global_step=step)
                        log('test acc...')

                        label,final_pred_label = sess.run([
                            model.y, model.preds],feed_dict=feed_dict)



                        log('label.shape           :'+str(label.shape)) # (batch_size , label_length)
                        log('final_pred_label.shape:'+str(np.asarray(final_pred_label).shape)) # (1, batch_size, decode_length<=label_length)

                        log('label           : '+str(label[0]))
                        log('final_pred_label: '+str( np.asarray(final_pred_label)[0]))


                    # TODO: serving
                    if args.serving :#and total_step // hp.steps_per_epoch > 5:
                        np.save('logdir/lm_pinyin_dict.npy',feeder.pny_vocab)
                        np.save('logdir/lm_hanzi_dict.npy',feeder.han_vocab)
                        print(total_step, 'hhhhhhhh')
                        # TODO: Set up serving builder and signature map
                        serve_dir = args.serving_dir + '0001'
                        if os.path.exists(serve_dir):
                            os.removedirs(serve_dir)
                        builder = tf.saved_model.builder.SavedModelBuilder(export_dir=serve_dir)
                        input = tf.saved_model.utils.build_tensor_info(model.x)
                        output_labels = tf.saved_model.utils.build_tensor_info(model.preds)

                        prediction_signature = (
                            tf.saved_model.signature_def_utils.build_signature_def(
                                inputs={'pinyin': input},
                                outputs={'hanzi': output_labels},
                                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
                        )
                        if first_serving:
                            first_serving = False
                            builder.add_meta_graph_and_variables(
                                sess=sess, tags=[tf.saved_model.tag_constants.SERVING],
                                signature_def_map={
                                    'predict_Pinyin2Hanzi':
                                        prediction_signature,
                                },
                                main_op=tf.tables_initializer(),
                                strip_default_attrs=True
                            )

                        builder.save()
                        log('Done store serving-model')
                        raise Exception('Done store serving-model')

                    # TODO: Validation
                    # if total_step % hp.steps_per_epoch == 0 and  i >= 10:
                    if total_step % hp.steps_per_epoch == 0:
                        log('validation...')
                        valid_start = time.time()
                        # TODO: validation
                        valid_hp = copy.deepcopy(hp)
                        print('feature_type: ',hp.feature_type)
                        valid_hp.data_type = 'dev'
                        valid_hp.thchs30 = True
                        valid_hp.aishell = True
                        valid_hp.prime = True
                        valid_hp.stcmd = True
                        valid_hp.shuffle = True
                        valid_hp.data_length = None

                        valid_feeder = DataFeeder(args=valid_hp)
                        valid_feeder.pny_vocab = feeder.pny_vocab
                        valid_feeder.han_vocab = feeder.han_vocab
                        # valid_feeder.am_vocab = feeder.am_vocab
                        valid_batch_data = valid_feeder.get_lm_batch()
                        log('valid_num_sentences:' + str(len(valid_feeder.wav_lst))) # 15219
                        valid_hp.input_vocab_size = len(valid_feeder.pny_vocab)
                        valid_hp.final_output_dim = len(valid_feeder.pny_vocab)
                        valid_hp.steps_per_epoch = len(valid_feeder.wav_lst) // valid_hp.batch_size
                        log('valid_steps_per_epoch:' + str(valid_hp.steps_per_epoch)) # 951
                        log('valid_pinyin_vocab_size:' + str(valid_hp.input_vocab_size)) # 1124
                        valid_hp.label_vocab_size = len(valid_feeder.han_vocab)
                        log('valid_label_vocab_size :' + str(valid_hp.label_vocab_size)) # 3327

                        # dev 只跑一个epoch就行
                        with tf.variable_scope('validation') as scope:
                            for k in range(len(valid_feeder.wav_lst) // valid_hp.batch_size):
                                valid_input_batch,valid_label_batch = next(valid_batch_data)
                                valid_feed_dict = {
                                    model.x: valid_input_batch,
                                    model.y: valid_label_batch,
                                }
                                # TODO: Run one step
                                valid_start_time = time.time()
                                valid_batch_loss,valid_batch_acc = sess.run([model.mean_loss,model.acc], feed_dict=valid_feed_dict)
                                valid_time_window.append(time.time() - valid_start_time)
                                valid_loss_window.append(valid_batch_loss)
                                valid_acc_window.append(valid_batch_acc)
                                # print('loss',loss,'batch_loss',batch_loss)
                                message = 'Valid-Step %-7d [%.03f sec/step, valid_loss=%.05f, avg_loss=%.05f, valid_acc=%.05f, avg_acc=%.05f]' % (
                                    valid_step, valid_time_window.average, valid_batch_loss, valid_loss_window.average,valid_batch_acc,valid_acc_window.average)
                                log(message)
                                summary_writer_dev.add_summary(sess.run(valid_stats,feed_dict=valid_feed_dict), valid_step)
                                valid_step += 1
                            log('Done Validation！Total Time Cost(sec):' + str(time.time()-valid_start))

        except Exception as e:
            log('Exiting due to exception: %s' % e)
            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser()
    # TODO: add arguments
    parser.add_argument('--log_dir', default=os.path.expanduser('~/my_asr2/logdir/logging'))
    parser.add_argument('--serving_dir', default=os.path.expanduser('~/my_asr2/logdir/serving_lm/'))
    parser.add_argument('--data_dir', default=os.path.expanduser('~/corpus_zn'))
    parser.add_argument('--model', default='ASR_transformer_encoder')
    parser.add_argument('--epochs', type=int, help='Max epochs to run.', default=10)
    parser.add_argument('--restore_step', type=int, help='Global step to restore from checkpoint.',default=93000)
    parser.add_argument('--serving', type=bool, help='', default=True)
    # parser.add_argument('--validation_interval', type=int, help='一个epoch验证5次，每次200步共3200条数据', default=7090) # 35450//5
    parser.add_argument('--summary_interval', type=int, default=10,help='Steps between running summary ops.')
    parser.add_argument('--checkpoint_interval', type=int, default=100, help='Steps between writing checkpoints.')
    parser.add_argument('--hparams', default='',
                        help='Hyperparameter overrides as a comma-separated list of name=value pairs')
    args = parser.parse_args()

    run_name = args.model
    log_dir = os.path.join(args.log_dir, 'logs-%s' % run_name)
    os.makedirs(log_dir, exist_ok=True)

    # TODO: launch init and train
    init(os.path.join(log_dir, 'train.log'), run_name)
    hp.parse(args.hparams)
    train(log_dir, args)

if __name__ == '__main__':
  main()