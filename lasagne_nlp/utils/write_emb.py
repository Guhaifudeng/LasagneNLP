# -*- coding: utf-8 -*-
# @Author: Jie
# @Date:   2017-02-15 17:36:37
# @Last Modified by:   Jie     @Contact: jieynlp@gmail.com
# @Last Modified time: 2017-02-20 17:42:37

import logging
import sys
import numpy as np
import lasagne
import gzip
import theano



def write_to_file(emb_list, output_file, init_emb_file= 'tmp/emb0'):
    word_list = []
    origin_emb_lines = open(init_emb_file,'r').readlines()
    for line in origin_emb_lines:
        word_list.append(line.strip().split()[0])

    emb_array = np.asarray(emb_list[0])
    # print "emb array len:", emb_array.shape
    assert emb_array.ndim == 2
    assert emb_array.shape[0] == len(word_list)
    out_file = open(output_file,'w')
    for idx in range(emb_array.shape[0]):
        out_file.write(word_list[idx]+ " ")
        for idy in range(emb_array.shape[1]):
            if idy == emb_array.shape[1] -1:
                out_file.write(str(emb_array[idx][idy])+"\n")
            else:
                out_file.write(str(emb_array[idx][idy])+" ")
    out_file.close()
            













