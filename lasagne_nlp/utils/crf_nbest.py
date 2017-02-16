# -*- coding: utf-8 -*-
# @Author: Jie
# @Date:   2017-02-15 17:36:37
# @Last Modified by:   Jie     @Contact: jieynlp@gmail.com
# @Last Modified time: 2017-02-16 09:44:28

import logging
import sys
import numpy as np
import lasagne
from gensim.models.word2vec import Word2Vec
import gzip
import theano


def get_logger(name, level=logging.INFO, handler=sys.stdout,
               formatter='%(asctime)s - %(name)s - %(levelname)s - %(message)s'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(formatter)
    stream_handler = logging.StreamHandler(handler)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger



def write_best(inputs, targets, masks, crf_para, label_alphabet, is_flattened=False):
    filename = 'tmp/best'
    batch_size = crf_para.shape[0]
    with open(filename, 'a') as file:
        for idx in range(batch_size):
            predict_label = generate_best(crf_para[idx], masks[idx])
            sent_length = len(predict_label)

            for idy in range(sent_length):
                file.write('_ %s %s\n' % (label_alphabet.get_instance(targets[idx, idy] + 1),
                                                  label_alphabet.get_instance(predict_label[idy]+1)))
            file.write('\n')



def generate_best(crf_para_one_sentence, masks_one_sentence):
    '''
    crf_para_one_sentence: [times, label_size, label_size], need to remove last label
    masks_one_sentence: [times,2]
    '''
    pi_time0 = crf_para_one_sentence[0,-1,:-1]
    crf_para_one_sentence = crf_para_one_sentence[:,:-1,:-1]
    sentence_length = -1
    for idx in range(masks_one_sentence.shape[0]):
        if masks_one_sentence[idx] > 0.:
            sentence_length = idx+1
    pointer_list = []
    for idx in range(sentence_length):
        if idx == 0:
            pi_timet = pi_time0
        else:
            pi_timet, pi_pointer = forword_score(pi_timet, crf_para_one_sentence[idx])
            pointer_list.append(pi_pointer.tolist())
    last_pointer = np.argmax(pi_timet,axis=0)
    label_id_list = [last_pointer]
    for idx in reversed(range(sentence_length-1)):
        last_pointer = pointer_list[idx][last_pointer]
        label_id_list.append(last_pointer)
    assert(len(label_id_list)== sentence_length)
    label_id_list.reverse()
    return label_id_list


def forword_score(pi_timet, crf_para_time):
    '''
    pi_timet: [label_size]
    '''
    assert pi_timet.ndim == 1
    assert crf_para_time.ndim == 2
    pi_timet_extend = np.full(crf_para_time.shape[1], pi_timet[0],dtype=pi_timet[0].dtype)
    for idx in range(1,pi_timet.shape[0]):
        pi_timet_extend= np.append(pi_timet_extend, np.full(crf_para_time.shape[1], pi_timet[idx],dtype=pi_timet[idx].dtype))
    pi_timet_extend = pi_timet_extend.reshape(crf_para_time.shape[0],crf_para_time.shape[1])
    # print pi_timet_extend
    pi_timet_extend = pi_timet_extend + crf_para_time
    pi_timet = np.amax(pi_timet_extend, axis=0)
    pi_pointer = np.argmax(pi_timet_extend, axis=0)
    # print pi_timet,pi_timet.shape, pi_pointer
    return pi_timet,pi_pointer



def write_nbest(inputs, targets, masks, crf_para, label_alphabet,filename, topn, is_flattened=False):
    batch_size = crf_para.shape[0]
    with open(filename, 'a') as file:
        for idx in range(batch_size):
            predict_label, last_pi = generate_nbest(crf_para[idx], masks[idx], topn)
            file.write("##score## ")
            ## write prob
            for idy in range(last_pi.shape[0]):
                if idy == last_pi.shape[0] - 1:
                    file.write(str(round(last_pi[idy],4))+'\n')
                else:
                    file.write(str(round(last_pi[idy],4))+' ')
            sent_length = len(predict_label)
            ## write nbest to file
            for idy in range(sent_length):
                file.write('_ %s ' % (label_alphabet.get_instance(targets[idx, idy] + 1)))
                for idz in range(len(predict_label[idy])):
                    if idz == len(predict_label[idy]) -1:
                        file.write(label_alphabet.get_instance(predict_label[idy][idz]+1)+"\n")
                    else:
                        file.write(label_alphabet.get_instance(predict_label[idy][idz]+1)+" ")
            file.write('\n')



def generate_nbest(crf_para_one_sentence, masks_one_sentence, topn):
    '''
    crf_para_one_sentence: [times, label_size, label_size], need to remove last label
    masks_one_sentence: [times,2]
    '''
    sentence_length = -1
    for idx in range(masks_one_sentence.shape[0]):
        if masks_one_sentence[idx] > 0.:
            sentence_length = idx+1
    return nbest_decode(crf_para_one_sentence,sentence_length, topn)


def nbest_decode(crf_para_one_sentence, sentence_length, topn):
    '''
    nbest = topn iff candidate > topn, else nbest = candidate
    crf_para_one_sentence: [times, label_size, label_size], need to remove last label
    pi_timet: [label,nbest] 
    preId_timet: [label, nbest]
    return returned_label_list: [sent_length, nbest], nbest in reversed order
    '''
    label_num = crf_para_one_sentence.shape[2]-1
    pi_list = []
    preId_list = []
    pi_timet = np.reshape(crf_para_one_sentence[0,-1,:-1],(label_num,1))
    preId_timet = np.full((label_num,1), -1, 'int64')
    crf_para_one_sentence = crf_para_one_sentence[:,:-1,:-1]
    pi_list.append(pi_timet.tolist())
    preId_list.append(preId_timet.tolist())
    for idx in range(1, sentence_length):
        pi_timet = np.asarray(pi_timet)
        candidate = pi_timet.shape[1]
        pi_timet_extend_nbest = []
        for idy in range(candidate):
            pi_timet_extend = np.repeat(np.reshape(pi_timet[:,idy],(label_num,1)),label_num, axis=1)
            pi_timet_extend = pi_timet_extend + crf_para_one_sentence[idx]
            # pi_timet_extend:[label,label]
            pi_timet_extend_nbest.append(pi_timet_extend.tolist())
        ## pi_timet_extend_nbest list to array, [nbest, label_in,label_out] to [label_out,nbest,label_in]
        ##
        pi_timet_extend_nbest = np.transpose(np.asarray(pi_timet_extend_nbest),(2,0,1))
        pi_timet_extend_nbest =np.reshape(pi_timet_extend_nbest, (label_num,-1))
        pi_timet = np.sort(pi_timet_extend_nbest, axis=1)[:,::-1]
        preId = np.argsort(pi_timet_extend_nbest, axis=1)[:,::-1]
        new_candidate = topn if pi_timet.shape[1] >= topn else pi_timet.shape[1]
        pi_timet = pi_timet[:,:new_candidate].tolist()
        preId = preId[:,:new_candidate]
        preId = preId_convert_list(preId, label_num)
        pi_list.append(pi_timet)
        preId_list.append(preId)
    ## last best score convert to [nbest,label] to fit function preId_convert_list_1D, then convert to 1d 
    last_best_score = np.ravel(np.transpose(np.asarray(pi_list[-1]),(1,0)))
    new_candidate = topn if last_best_score.shape[0] >= topn else last_best_score.shape[0]
    last_pi = np.sort(last_best_score)[::-1][:new_candidate]
    last_Id = np.argsort(last_best_score)[::-1][:new_candidate]
    last_Id = preId_convert_list_1D(last_Id, label_num)

    ## back trace 
    label_time_index, nbest_time_index = recover_comp_string_to_index(last_Id)
    label_list = [label_time_index]

    for idx in reversed(range(1,sentence_length)):
        nbest = len(label_time_index)
        nbest_comp_list = []
        for idy in range(nbest):
            nbest_comp_list.append(preId_list[idx][label_time_index[idy]][nbest_time_index[idy]])
        label_time_index, nbest_time_index = recover_comp_string_to_index(nbest_comp_list)
        label_list.append(label_time_index)
    ## reverse label_list in both two dims:
    #  1.reverse first dim is to reorder label from begin to end
    #  2. reverse second dim is to keep best candidate in the last column when write to file
    returned_label_list = []
    for idx in reversed(range(len(label_list))):
        reversed_label = label_list[idx][::-1]
        returned_label_list.append(reversed_label)
    ## return label list and prob, reverse pi to keep best candidate in last column, prob using softmax, 
    return returned_label_list, softmax(last_pi[::-1])



def preId_convert_list(preId, label_num):
    assert preId.shape[0] == label_num
    new_preId = []
    for idx in range(preId.shape[0]):
        temp = []
        oneD_list = preId_convert_list_1D(preId[idx],label_num)
        new_preId.append(oneD_list)
    return new_preId


def preId_convert_list_1D(preId,label_num):
    assert preId.ndim == 1
    oneD_list = []
    for idx in range(preId.shape[0]):
        pre_label = preId[idx]%label_num
        pre_nbest = preId[idx]/label_num
        oneD_list.append(str(pre_label)+"@"+str(pre_nbest))
    return oneD_list


def recover_comp_string_to_index(comp_string_list):
    label_index = []
    nbest_index = []
    for comp_string in comp_string_list:
        label_index.append(int(comp_string.split('@')[0]))
        nbest_index.append(int(comp_string.split('@')[1]))
    return label_index, nbest_index


def softmax(oneD_array):
    if np.isnan(oneD_array).any():
        print "x nan:",oneD_array
    e_x = np.exp(oneD_array)
    if np.isnan(e_x).any():
        print "e_x nan:",e_x
    return e_x/e_x.sum()



def back_onebest_decode(crf_para_one_sentence, sentence_length, topn):
    '''
    crf_para_one_sentence: [times, label_size, label_size], need to remove last label
    pi_timet: [label,nbest], nbest = topn iff candidate > topn, else nbest = candidate
    preId_timet: [label, nbest], nbest = topn iff candidate > topn, else nbest = candidate
    '''
    label_num = crf_para_one_sentence.shape[2]-1
    pi_list = []
    preId_list = []
    pi_timet = np.reshape(crf_para_one_sentence[0,-1,:-1],(label_num,1))
    preId_timet = np.full((label_num,1), -1, 'int64')
    crf_para_one_sentence = crf_para_one_sentence[:,:-1,:-1]
    pi_list.append(pi_timet.tolist())
    preId_list.append(preId_timet.tolist())
    for idx in range(1, sentence_length):
        pi_timet = np.asarray(pi_timet)
        candidate = pi_timet.shape[1]
        for idy in range(candidate):
            pi_timet_extend = np.repeat(np.reshape(pi_timet[:,idy],(label_num,1)),label_num, axis=1)
            pi_timet_extend = pi_timet_extend + crf_para_one_sentence[idx]
            # pi_timet_extend:[label,label]
            pi_timet = np.amax(pi_timet_extend, axis=0)
            preId = np.argmax(pi_timet_extend, axis=0)
            pi_timet = np.reshape(pi_timet, (label_num,1)).tolist()
            pi_list.append(pi_timet)
            preId_list.append(np.reshape(preId, (label_num,1)).tolist())

    last_pi = np.asarray(pi_list)[-1]
    last_Id = np.argmax(last_pi, axis=0)
    ## back source 
    label_list = [last_Id.tolist()]
    for idx in reversed(range(1,sentence_length)):
        last_Id = preId_list[idx][last_Id][0]
        label_list.append([last_Id])
    label_list.reverse()
    return label_list














