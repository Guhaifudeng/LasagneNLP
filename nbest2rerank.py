#!/usr/bin/env python
# coding=utf-8

##===================================================================##
#   Data preprocessing for Neural rerank with base CRF++0.58  
#   Jie Yang
#   May. 1, 2015
# 
##===================================================================##
import sys
import os
import types
import re
import math
import numpy as np

label_set = "BIO"

def nbest2rerank_collapseLabel(input_file, add_best=False):
    print "Start to turn Nbest format to rerank format for file: ", input_file
    in_lines = open(input_file,'rU').readlines()
    word_list = []
    label_list = []
    total_gold = 0
    out_file = open(input_file,'w')
    sent_num = 0
    add_gold_index = 0
    new_split = 0
    split_index = 0
    for line in in_lines:
        if "##score##" in line:
            prob_list = line.strip('\n').split(' ')[1:]
            continue
        elif len(line) < 2:
            split_index += 1
            label_list = np.transpose(np.asarray(label_list), (1,0)).tolist()
            nbest = len(label_list) - 1
            assert nbest == len(prob_list)
            GoldExist = False
            for idx in reversed(range(0-nbest,0)):
                ifGold = writeCollapseSentence(out_file, word_list, label_list[idx], label_list[0], prob_list[idx])
                new_split += 1
                if ifGold:
                    if GoldExist:
                        print "double gold found:",word_list
                        assert GoldExist == False
                    GoldExist = True
                    total_gold += 1
            ## if gold not exist and add best, add best in
            if (not GoldExist) and add_best:
                new_split += 1
                # print "Can't find gold list, add gold,",add_gold_index
                # print word_list
                add_gold_index += 1
                ifGold = writeCollapseSentence(out_file, word_list, label_list[0], label_list[0], prob_list[0])
                assert ifGold == True
            word_list = []
            label_list = []
            sent_num += 1
        else:
            pairs = line.strip('\n').split(' ')
            word_list.append(pairs[0])
            label_list.append(pairs[1:])
    print "convert to rerank format! Total sent num:", sent_num
    print "Gold exist ratio:",(total_gold+0.)/sent_num


def collapseEachSentence(single_sentence_list, pred_label_list, gold_label_list):
    sentence_length = len(single_sentence_list)
    ## label in upper format
    if (sentence_length != len(pred_label_list))|(sentence_length != len(gold_label_list)):
        print "Error in collapseEachSentence! Sentence, label and gold size mismatch!"
        return
    matched_label_size = 0
    for idx in range(len(gold_label_list)):
        if pred_label_list[idx] == gold_label_list[idx]:
            matched_label_size += 1
    precision_value = (matched_label_size+0.0)/sentence_length
    if label_set == "BMES":
        pred_ner = get_ner_BMES(pred_label_list)
        gold_ner = get_ner_BMES(gold_label_list)
    elif label_set == "BIO":
        pred_ner = get_ner_BIO(pred_label_list)
        gold_ner = get_ner_BIO(gold_label_list)
    else:
        print "ERROR: label set not BIO or BMES"

    matched_ner = list(set(pred_ner)&set(gold_ner))
    if (len(pred_ner) == 0) and (len(gold_ner) == 0):
        f1_measure = 1.
    elif len(pred_ner) == 0:
        f1_measure = 0.
    elif len(gold_ner) == 0:
        f1_measure = 0.
    else:
        p = (len(matched_ner) +0.)/len(pred_ner)
        r = (len(matched_ner) +0.)/len(gold_ner)
        if (p + r)== 0:
            f1_measure = 0.
        else:
            f1_measure = 2*p*r/(p+r)

    collapseSentence = []
    collapseOrigin = []
    collapseGold = []
    collapsePredict = []
    span_list, catagory_list = ner_span(sentence_length,pred_ner)
    assert len(span_list) == len(catagory_list)
    collSymbol = '*#*'
    for idx in range(len(catagory_list)):
        if catagory_list[idx] == "O":
            for idy in span_list[idx]:
                collapseSentence.append(single_sentence_list[idy])
                collapseOrigin.append(single_sentence_list[idy])
                collapseGold.append(gold_label_list[idy])
                collapsePredict.append(pred_label_list[idy])
        else:
            collWord = '#'+catagory_list[idx]+ '#'
            collOrg = ''
            collGold = ''
            collPred = ''
            for idy in span_list[idx]:
                collOrg += single_sentence_list[idy] + collSymbol
                collGold += gold_label_list[idy] + collSymbol
                collPred += pred_label_list[idy] + collSymbol
            symbol_len = len(collSymbol)
            collapseSentence.append(collWord)
            collapseOrigin.append(collOrg[:-symbol_len])
            collapseGold.append(collGold[:-symbol_len])
            collapsePredict.append(collPred[:-symbol_len])

    if ((len(collapseSentence)!=len(collapseOrigin))|(len(collapseSentence)!=len(collapseGold))|(len(collapseSentence)!=len(collapsePredict))):
        print "Error in collapseEachSentence! Generate different size lists!"
        return
    return collapseSentence, collapseOrigin, collapseGold, collapsePredict, precision_value, f1_measure


def ner_span(sentence_length, ner_list):
    ### ner_list is like [[0]ORG, [3,5]PER]
    ### return two list, one is the span list, other is catagory list
    ner_catagory_list = []
    ner_span_list = []
    for ner in ner_list:
        pair = ner.strip('[').split(']')
        ner_catagory_list.append(pair[1])
        if "," in pair[0]:
            pos = pair[0].split(',')
            span = range(int(pos[0]),int(pos[1])+1)
            ner_span_list.append(span)
        else:
            ner_span_list.append([int(pair[0])])
    catagory_list = []
    span_list = []
    for idx in range(len(ner_span_list)):
        span = ner_span_list[idx]
        cata = ner_catagory_list[idx]
        if idx == 0:
            if span[0] > 0:
                catagory_list.append('O')
                span_list.append(range(span[0]))
            catagory_list.append(cata)
            span_list.append(span)
        else:
            if span[0] > span_list[-1][-1] + 1:
                catagory_list.append('O')
                span_list.append(range(span_list[-1][-1] + 1, span[0]))
            catagory_list.append(cata)
            span_list.append(span)
    if len(span_list) > 0:
        if span_list[-1][-1] !=  sentence_length-1:
            catagory_list.append('O')
            span_list.append(range(span_list[-1][-1] + 1, sentence_length))
    else:
        catagory_list.append('O')
        span_list.append(range(0, sentence_length))
    return span_list, catagory_list





def writeCollapseSentence(out_file, single_sentence_list, the_label_list, gold_label_list, probability):
    collapseSentence, collapseOrigin, collapseGold, collapsePredict, precision_value, f1_measure = collapseEachSentence(single_sentence_list, the_label_list, gold_label_list)
    if precision_value == 1.:
        ifGold = True
    else:
        ifGold = False
    for idx in range(0,len(collapseSentence)):
        out_file.write(collapseSentence[idx] + ' [D]'+collapseOrigin[idx] + ' [G]' + collapseGold[idx] + ' [P]' + collapsePredict[idx] +' [V]'+ str(round(precision_value,4))+ ' [F]'+ str(round(f1_measure,4))+ ' [T]' + float2class(probability)+ ' [R]' + probability + ' ' + str(ifGold) +'\n')
        # out_file.write(collapseSentence[idx] + ' [D]'+collapseOrigin[idx] + ' [G]' + collapseGold[idx] + ' [P]' + collapsePredict[idx] + ' [T]' + float2class(probability)+ ' [R]' + probability + ' ' +judge +'\n')
    out_file.write('\n')
    return ifGold


def float2class(input_number):
    class_size = 10
    float_number = float(input_number)
    if float_number > 0 :
        for idx in range(-1*(class_size-2), 1):
            if math.log(float_number) < idx:
                return str(idx)
    else:
        return str(-class_size)
    return str(class_size)


def get_ner_BMES(label_list):
    # list_len = len(word_list)
    # assert(list_len == len(label_list)), "word list size unmatch with label list"
    list_len = len(label_list)
    begin_label = 'B-'
    end_label = 'E-'
    single_label = 'S-'
    whole_tag = ''
    index_tag = ''
    tag_list = []
    stand_matrix = []
    for i in range(0, list_len):
        # wordlabel = word_list[i]
        current_label = label_list[i].upper()
        if begin_label in current_label:
            if index_tag != '':
                tag_list.append(whole_tag + ',' + str(i-1))
            whole_tag = current_label.replace(begin_label,"",1) +'[' +str(i)
            index_tag = current_label.replace(begin_label,"",1)
            
        elif single_label in current_label:
            if index_tag != '':
                tag_list.append(whole_tag + ',' + str(i-1))
            whole_tag = current_label.replace(single_label,"",1) +'[' +str(i)
            tag_list.append(whole_tag)
            whole_tag = ""
            index_tag = ""
        elif end_label in current_label:
            if index_tag != '':
                tag_list.append(whole_tag +',' + str(i))
            whole_tag = ''
            index_tag = ''
        else:
            continue
    if (whole_tag != '')&(index_tag != ''):
        tag_list.append(whole_tag)
    tag_list_len = len(tag_list)

    for i in range(0, tag_list_len):
        if  len(tag_list[i]) > 0:
            tag_list[i] = tag_list[i]+ ']'
            insert_list = reverse_style(tag_list[i])
            stand_matrix.append(insert_list)
    # print stand_matrix
    return stand_matrix


def get_ner_BIO(label_list):
    # list_len = len(word_list)
    # assert(list_len == len(label_list)), "word list size unmatch with label list"
    list_len = len(label_list)
    begin_label = 'B-'
    inside_label = 'I-' 
    whole_tag = ''
    index_tag = ''
    tag_list = []
    stand_matrix = []
    for i in range(0, list_len):
        # wordlabel = word_list[i]
        current_label = label_list[i].upper()
        if begin_label in current_label:
            if index_tag == '':
                whole_tag = current_label.replace(begin_label,"",1) +'[' +str(i)
                index_tag = current_label.replace(begin_label,"",1)
            else:
                tag_list.append(whole_tag + ',' + str(i-1))
                whole_tag = current_label.replace(begin_label,"",1)  + '[' + str(i)
                index_tag = current_label.replace(begin_label,"",1)

        elif inside_label in current_label:
            if current_label.replace(inside_label,"",1) == index_tag:
                whole_tag = whole_tag 
            else:
                if (whole_tag != '')&(index_tag != ''):
                    tag_list.append(whole_tag +',' + str(i-1))
                whole_tag = ''
                index_tag = ''
        else:
            if (whole_tag != '')&(index_tag != ''):
                tag_list.append(whole_tag +',' + str(i-1))
            whole_tag = ''
            index_tag = ''

    if (whole_tag != '')&(index_tag != ''):
        tag_list.append(whole_tag)
    tag_list_len = len(tag_list)

    for i in range(0, tag_list_len):
        if  len(tag_list[i]) > 0:
            tag_list[i] = tag_list[i]+ ']'
            insert_list = reverse_style(tag_list[i])
            stand_matrix.append(insert_list)
    return stand_matrix



def reverse_style(input_string):
    target_position = input_string.index('[')
    input_len = len(input_string)
    output_string = input_string[target_position:input_len] + input_string[0:target_position]
    return output_string



def insert_word_to_nbest(base_file, nbest_file, output_file):
    nbest_lines = open(nbest_file,'r').readlines()
    base_lines = open(base_file,'r').readlines()
    out_file = open(output_file,'w')
    base_index = 0
    for idx in range(len(nbest_lines)):
        if "##score##" in nbest_lines[idx]:
            out_file.write(nbest_lines[idx])
            continue
        elif len(nbest_lines[idx]) < 2:
            out_file.write(nbest_lines[idx])
            base_index += 1
            continue
        else:
            nbest_pair = nbest_lines[idx].split(' ',1)
            base_pair = base_lines[base_index].split(' ',1)
            nbest_lines[idx] = base_pair[0] + ' ' + nbest_pair[1]
            out_file.write(nbest_lines[idx])
        base_index += 1
    #print "base index:",base_index
    #print "base line num:", len(base_lines)
    assert base_index == len(base_lines)
    out_file.close()




if __name__ == '__main__':
    # nbest2rerank_collapseLabel('tmp/nbest_debug.dev.bioes2', True)
    # output_file = 'tmp/nbest_debug.dev.rerank'
    # insert_word_to_nbest('data/debug.dev.bioes','tmp/nbest_debug.dev.bioes2', output_file)
    # nbest2rerank_collapseLabel(output_file, False)
    output_file = sys.argv[3]
    insert_word_to_nbest(sys.argv[1], sys.argv[2], output_file)
    nbest2rerank_collapseLabel(output_file, True)

    
