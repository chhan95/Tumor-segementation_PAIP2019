
import io
import itertools
import json
import os
import random
import re
import shutil
import textwrap

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from termcolor import colored


####
def check_manual_seed(seed):
    """ 
    If manual seed is not specified, choose a random one and notify it to the user.
    """

    seed = seed or random.randint(1, 10000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    print('Using manual seed: {seed}'.format(seed=seed))
    return

####
def check_log_dir(log_dir):
    # check if log dir exist
    if os.path.isdir(log_dir):
        color_word = colored('WARNING', color='red', attrs=['bold', 'blink'])
        print('%s: %s exist!' % (color_word, colored(log_dir, attrs=['underline'])))
        while (True):
            print('Select Action: d (delete) / q (quit)', end='')
            key = input()
            if key == 'd':
                shutil.rmtree(log_dir)
                break
            elif key == 'q':
                exit()
            else:
                color_word = colored('ERR', color='red')
                print('---[%s] Unrecognize Characters!' % color_word)
    return

####
def plot_confusion_matrix(conf_mat, labels):
    ''' 
    Parameters:
        title='Confusion matrix'        : Title for your matrix
        tensor_name = 'MyFigure/image'  : Name for the output summay tensor

    Returns:
        summary: image of plot figure

    Other itema to note:
        - Depending on the number of category and the data , you may have to modify the figzie, font sizes etc. 
        - Currently, some of the ticks dont line up due to rotations.
    '''
    cm = conf_mat

    np.set_printoptions(precision=2)
    ###

    fig = matplotlib.figure.Figure(figsize=(7, 7), dpi=320, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(cm, cmap='Oranges')

    classes = [re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', x) for x in labels]
    classes = ['\n'.join(textwrap.wrap(l, 40)) for l in classes]

    tick_marks = np.arange(len(classes))

    ax.set_xlabel('Predicted', fontsize=7)
    ax.set_xticks(tick_marks)
    c = ax.set_xticklabels(classes, fontsize=4, rotation=-90,  ha='center')
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()

    ax.set_ylabel('True Label', fontsize=7)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=4, va ='center')
    ax.yaxis.set_label_position('left')
    ax.yaxis.tick_left()

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], 'd') if cm[i,j]!=0 else '.', 
            horizontalalignment="center", fontsize=6, 
            verticalalignment='center', color= "black")
    fig.set_tight_layout(True)

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()

    # get PNG data from the figure
    png_buffer = io.BytesIO()
    fig.canvas.print_png(png_buffer)
    png_encoded = png_buffer.getvalue()
    png_buffer.close()

    return png_encoded
####
def update_log(output, epoch, prefix, color, tfwriter, log_file, logging):

    # print values and convert
    max_length = len(max(output.keys(), key=len))
    for metric in output:
        key = colored(prefix + '-' + metric.ljust(max_length), color)
        print('------%s : ' % key, end='')
        if metric != 'conf_mat':
            print('%0.7f' % output[metric])
        else:
            conf_mat = output['conf_mat'] # use pivot to turn back
            conf_mat_df = pd.DataFrame(conf_mat)
            conf_mat_df.index.name = 'True'
            conf_mat_df.columns.name = 'Pred'
            output['conf_mat'] = conf_mat_df
            print('\n', conf_mat_df)

    if not logging:
        return

    # create stat dicts
    stat_dict = {}
    for metric in output:
        if metric != 'conf_mat':
            metric_value = output[metric] 
        else:
            conf_mat_df = output['conf_mat'] # use pivot to turn back
            conf_mat_df = conf_mat_df.unstack().rename('value').reset_index()
            conf_mat_df = pd.Series({'conf_mat' : conf_mat}).to_json(orient='records')
            metric_value = conf_mat_df
        stat_dict['%s-%s' % (prefix, metric)] = metric_value

    # json stat log file, update and overwrite
    with open(log_file) as json_file:
        json_data = json.load(json_file)

    current_epoch = str(epoch)
    if current_epoch in json_data:
        old_stat_dict = json_data[current_epoch]
        stat_dict.update(old_stat_dict)
    current_epoch_dict = {current_epoch : stat_dict}
    json_data.update(current_epoch_dict)

    with open(log_file, 'w') as json_file:
        json.dump(json_data, json_file)

    # log values to tensorboard
    for metric in output:
        if metric != 'conf_mat':
            tfwriter.add_scalar(prefix + '-' + metric, output[metric], current_epoch)

####
def log_train_ema_results(engine, info):
    """
    running training measurement
    """
    training_ema_output = engine.state.metrics #
    training_ema_output['lr'] = float(info['optimizer'].param_groups[0]['lr'])
    update_log(training_ema_output, engine.state.epoch, 'train-ema', 'green',
                info['tfwriter'], info['json_file'], info['logging'])

####
def process_accumulated_output(output, batch_size, nr_classes):
    #
    def uneven_seq_to_np(seq):
        item_count = batch_size * (len(seq) - 1) + len(seq[-1])
        cat_array = np.zeros((item_count,) + seq[0][0].shape, seq[0].dtype)
        # BUG: odd len even
        for idx in range(0, len(seq)-1):
            cat_array[idx   * batch_size : 
                    (idx+1) * batch_size] = seq[idx] 
        cat_array[(idx+1) * batch_size:] = seq[-1]
        return cat_array
    #
    prob = uneven_seq_to_np(output['prob'])
    true = uneven_seq_to_np(output['true'])
    # threshold then get accuracy
    pred = np.argmax(prob, axis=-1)
    acc = np.mean(pred == true)
    # confusion matrix
    conf_mat = confusion_matrix(true, pred, 
                        labels=np.arange(nr_classes))
    #
    proc_output = dict(acc=acc, conf_mat=conf_mat)
    return proc_output

####
def inference(engine, inferer, dataloader, info):
    """
    inference measurement
    """
    inferer.accumulator = {metric : [] for metric in info['metric_names']}
    inferer.run(dataloader)
    output_stat = process_accumulated_output(inferer.accumulator, 
                                info['infer_batch_size'], info['nr_classes'])
    update_log(output_stat, engine.state.epoch, 'valid', 'red', 
                info['tfwriter'], info['json_file'], info['logging'])
    return

####
def accumulate_outputs(engine):
    batch_output = engine.state.output
    for key, item in batch_output.items():
        engine.accumulator[key].extend([item])
    return
