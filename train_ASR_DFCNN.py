import tensorflow as tf
import argparse
import os,math,copy
from utils.mylogger import *
from utils import ValueWindow , plot
from hparams import hparams as hp
from hparams import hparams_debug_string
from datafeeder import DataFeeder,GetEditDistance
from models import create_model
import datetime,time
import traceback,random
import numpy as np
from keras import backend as K


def add_stats(model):
  with tf.variable_scope('stats') as scope:
    tf.summary.histogram('pred_logits', model.pred_logits)
    tf.summary.histogram('labels', model.labels)
    tf.summary.scalar('batch_loss', model.batch_loss)
    tf.summary.scalar('learning_rate', model.learning_rate)
    gradient_norms = [tf.norm(grad) for grad in model.gradients]
    tf.summary.histogram('gradient_norm', gradient_norms)
    tf.summary.scalar('max_gradient_norm', tf.reduce_max(gradient_norms))
    return tf.summary.merge_all()

def add_dev_stats(model):
    with tf.variable_scope('dev_stats') as scope:
        tf.summary.scalar('dev_batch_loss',model.batch_loss)
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
        hp.data_length=None
        hp.decay_learning_rate= False
        hp.initial_learning_rate = 0.0005
        feeder = DataFeeder(args=hp)
        log('num_wavs:'+str(len(feeder.wav_lst))) # 283600
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
        model.add_decoder()
        model.add_optimizer(global_step=global_step)
        # TODO: summary
        stats = add_stats(model=model)
        valid_stats = add_dev_stats(model)


    # TODO：Set up saver and Bookkeeping
    time_window = ValueWindow(100)
    loss_window = ValueWindow(100)
    # wer_window = ValueWindow(100)
    valid_time_window = ValueWindow(100)
    valid_loss_window = ValueWindow(100)
    valid_wer_window = ValueWindow(100)
    saver = tf.train.Saver(max_to_keep=20)
    first_serving = True
    # TODO: train
    with tf.Session() as sess:
        log(hparams_debug_string(hp))
        try:
            # TODO: Set writer and initializer
            summary_writer = tf.summary.FileWriter(logdir, sess.graph)
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
                batch_data = feeder.get_am_batch()
                log('Traning epoch '+ str(i)+':')
                for j in range(hp.steps_per_epoch):
                    input_batch = next(batch_data)
                    feed_dict = {model.inputs:input_batch['the_inputs'],
                                 model.labels:input_batch['the_labels'],
                                 model.input_lengths:input_batch['input_length'],
                                 model.label_lengths:input_batch['label_length']}
                    # TODO: Run one step
                    start_time = time.time()
                    total_step, array_loss, batch_loss,opt = sess.run([global_step, model.ctc_loss,
                                                            model.batch_loss,model.optimize],feed_dict=feed_dict)
                    time_window.append(time.time() - start_time)
                    step = total_step

                    # TODO: Append loss
                    # loss = np.sum(array_loss).item()/hp.batch_size
                    loss_window.append(batch_loss)

                    # print('loss',loss,'batch_loss',batch_loss)
                    message = 'Step %-7d [%.03f sec/step, loss=%.05f, avg_loss=%.05f, lr=%.07f]' % (
                        step, time_window.average, batch_loss, loss_window.average,K.get_value(model.learning_rate))
                    log(message)
                    # ctcloss返回值是[batch_size,1]形式的所以刚开始没sum报错only size-1 arrays can be converted to Python scalars

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
                        # eval_start_time = time.time()
                        # with tf.name_scope('eval') as scope:
                        #     with open(os.path.expanduser('~/my_asr2/datasets/resource/preprocessedData/dev-meta.txt'), encoding='utf-8') as f:
                        #         metadata = [line.strip().split('|') for line in f]
                        #         random.shuffle(metadata)
                        #     eval_loss = []
                        #     batch_size = args.hp.batch_size
                        #     batchs = len(metadata)//batch_size
                        #     for i in range(batchs):
                        #         batch = metadata[i*batch_size : i*batch_size+batch_size]
                        #         batch = list(map(eval_get_example,batch))
                        #         batch = eval_prepare_batch(batch)
                        #         feed_dict = {'labels':batch[0],}
                        #     label,final_pred_label ,log_probabilities = sess.run([
                        #         model.labels[0], model.decoded[0], model.log_probabilities[0]])
                        #     # 刚开始没有加[]会报错 https://github.com/tensorflow/tensorflow/issues/11840
                        #     print('label:            ' ,label)
                        #     print('final_pred_label: ', final_pred_label[0])
                        #     log('eval time: %.03f, avg_eval_loss: %.05f' % (time.time()-eval_start_time,np.mean(eval_loss)))

                        label,final_pred_label ,log_probabilities,y_pred2 = sess.run([
                            model.labels, model.decoded, model.log_probabilities,model.y_pred2],feed_dict=feed_dict)
                        # 刚开始没有加[]会报错 https://github.com/tensorflow/tensorflow/issues/11840
                        log('label.shape           :'+str(label.shape)) # (batch_size , label_length)
                        log('final_pred_label.shape:'+str(np.asarray(final_pred_label).shape)) # (1, batch_size, decode_length<=label_length)
                        log('y_pred2.shape         : '+str( y_pred2.shape))
                        log('label:            ' +str(label[0]))
                        log('y_pred2         : '+str(y_pred2[0]))
                        log('final_pred_label: '+str(np.asarray(final_pred_label)[0][0]))

                        #  刚开始打不出来，因为使用的tf.nn.ctc_beam_decoder，这个返回的是sparse tensor所以打不出来
                        #  后来用keras封装的decoder自动将sparse转成dense tensor才能打出来

                        # waveform = audio.inv_spectrogram(spectrogram.T)
                        # audio.save_wav(waveform, os.path.join(logdir, 'step-%d-audio.wav' % step))
                        # plot.plot_alignment(alignment, os.path.join(logdir, 'step-%d-align.png' % step),
                        #                     info='%s, %s, %s, step=%d, loss=%.5f' % (
                        #                     args.model, commit, time_string(), step, loss))
                        # log('Input: %s' % sequence_to_text(input_seq))

                    # TODO: Check stop
                    if step % hp.steps_per_epoch ==0:
                        # TODO: Set up serving builder and signature map
                        serve_dir = args.serving_dir + '_' + str(total_step//hp.steps_per_epoch -1)
                        if os.path.exists(serve_dir):
                            os.removedirs(serve_dir)
                        builder = tf.saved_model.builder.SavedModelBuilder(export_dir=serve_dir)
                        input_spec = tf.saved_model.utils.build_tensor_info(model.inputs)
                        input_len = tf.saved_model.utils.build_tensor_info(model.input_lengths)
                        output_labels = tf.saved_model.utils.build_tensor_info(model.decoded2)
                        output_logits = tf.saved_model.utils.build_tensor_info(model.pred_logits)
                        prediction_signature = (
                            tf.saved_model.signature_def_utils.build_signature_def(
                                inputs={'spec': input_spec, 'len': input_len},
                                outputs={'label': output_labels, 'logits': output_logits},
                                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
                        )
                        if first_serving:
                            first_serving = False
                            builder.add_meta_graph_and_variables(
                                sess=sess, tags=[tf.saved_model.tag_constants.SERVING, 'ASR'],
                                signature_def_map={
                                    'predict_AudioSpec2Pinyin':
                                        prediction_signature,
                                },
                                main_op=tf.tables_initializer(),
                                strip_default_attrs=True
                            )
                        else:
                            builder.add_meta_graph_and_variables(
                                sess=sess, tags=[tf.saved_model.tag_constants.SERVING, 'ASR'],
                                signature_def_map={
                                    'predict_AudioSpec2Pinyin':
                                        prediction_signature,
                                },
                                strip_default_attrs=True
                            )
                        builder.save()
                        log('Done store serving-model')

                    # TODO: Validation
                    if step % hp.steps_per_epoch == 0 :
                        log('validation...')
                        valid_start = time.time()
                        # TODO: validation
                        valid_hp = copy.deepcopy(hp)
                        valid_hp.data_type = 'dev'
                        valid_hp.thchs30 = True
                        valid_hp.aishell = True
                        valid_hp.prime = False
                        valid_hp.stcmd = False
                        valid_hp.shuffle = True
                        valid_hp.data_length = None

                        valid_feeder = DataFeeder(args=valid_hp)
                        valid_batch_data = valid_feeder.get_am_batch()
                        log('valid_num_wavs:' + str(len(valid_feeder.wav_lst))) # 15219
                        valid_hp.input_vocab_size = len(valid_feeder.pny_vocab)
                        valid_hp.final_output_dim = len(valid_feeder.pny_vocab)
                        valid_hp.steps_per_epoch = len(valid_feeder.wav_lst) // valid_hp.batch_size
                        log('valid_steps_per_epoch:' + str(valid_hp.steps_per_epoch)) # 951
                        log('valid_pinyin_vocab_size:' + str(valid_hp.input_vocab_size)) # 1124
                        valid_hp.label_vocab_size = len(valid_feeder.han_vocab)
                        log('valid_label_vocab_size :' + str(valid_hp.label_vocab_size)) # 3327
                        words_num = 0
                        word_error_num = 0

                        # dev 只跑一个epoch就行
                        with tf.variable_scope('validation') as scope:
                            for k in range(len(valid_feeder.wav_lst) // valid_hp.batch_size):
                                valid_input_batch = next(valid_batch_data)
                                valid_feed_dict = {model.inputs: valid_input_batch['the_inputs'],
                                             model.labels: valid_input_batch['the_labels'],
                                             model.input_lengths: valid_input_batch['input_length'],
                                             model.label_lengths: valid_input_batch['label_length']}
                                # TODO: Run one step
                                valid_start_time = time.time()
                                valid_batch_loss ,valid_WER = sess.run([model.batch_loss,model.WER], feed_dict=valid_feed_dict)
                                valid_time_window.append(time.time() - valid_start_time)
                                valid_loss_window.append(valid_batch_loss)

                                # print('loss',loss,'batch_loss',batch_loss)
                                message = 'Valid-Step %-7d [%.03f sec/step, valid_loss=%.05f, avg_loss=%.05f,  WER=%.05f, avg_WER=%.05f]' % (
                                    valid_step, valid_time_window.average, valid_batch_loss, valid_loss_window.average,valid_WER,valid_wer_window.average)
                                log(message)
                                summary_writer.add_summary(sess.run(valid_stats,feed_dict=valid_feed_dict), valid_step)
                                valid_step += 1
                            log('Done Validation！Total Time Cost(sec):' + str(time.time()-valid_start))

        except Exception as e:
            log('Exiting due to exception: %s' % e)
            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser()
    # TODO: add arguments
    parser.add_argument('--log_dir', default=os.path.expanduser('~/my_asr2/logdir/logging'))
    parser.add_argument('--serving_dir', default=os.path.expanduser('~/my_asr2/logdir/serving_asr_0331'))
    parser.add_argument('--data_dir', default=os.path.expanduser('~/corpus_zn'))
    parser.add_argument('--model', default='ASR')
    parser.add_argument('--epochs', type=int, help='Max epochs to run.', default=100)
    parser.add_argument('--restore_step', type=int, help='Global step to restore from checkpoint.',default=16800)
    # parser.add_argument('--serving_interval', type=int, help='', default=35450)
    # parser.add_argument('--validation_interval', type=int, help='一个epoch验证5次，每次200步共3200条数据', default=30000) # 35450//5
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