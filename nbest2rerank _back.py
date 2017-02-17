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


def clear_char(input_file):
    in_lines = open(input_file,'rU').readlines()
    out_file = open(input_file,'w')
    for line in in_lines:
        if len(line) < 2:
            out_file.write(line)
        else:
            new_line = ""
            feature_list = line.split(' ')
            for element in feature_list:
                if "[C]" in element:
                    continue
                elif "[T]" in element:
                    continue
                else:
                    if new_line != '':
                        new_line = new_line + ' ' + element
                    else:
                        new_line = element
            out_file.write(new_line)
    out_file.close()

def convert2conll(input_file):
    in_lines = open(input_file,'rU').readlines()
    out_file = open(input_file,'w')
    for line in in_lines:
        if len(line) < 2:
            out_file.write(line)

        elif line[0]=='#':
            out_file.write(line)
        else:
            new_line = ""
            feature_list = line.split('\t')
            
            for idx in range(0,len(feature_list)):
                if idx == 0:
                    new_line = feature_list[0] + ' POS'
                    continue
                if "[S]" in feature_list[idx]:
                    continue
                else:
                    new_line = new_line + ' ' + feature_list[idx]
            
            out_file.write(new_line)
    out_file.close()        


def split_data(train_data, number):
    number = int(number)
    print "Start spliting data: ", train_data, " into ",number, "parts"
    in_lines = open(train_data,'rU').readlines()
    sentence_list = []
    sentence = []

    for line in in_lines:
        if len(line) > 3:
            sentence.append(line)
        else:
            sentence_list.append(sentence)
            sentence = []

    block_num = len(sentence_list)/number

    for idx in range(0,number):
        start_pos = idx * block_num
        end_pos = (idx+1)*block_num
        if (idx == number-1):
            end_pos = len(sentence_list)
        out_file = open(train_data + str(idx), 'w')
        for idy in range(start_pos,end_pos):
            for word in sentence_list[idy]:
                out_file.write(word)
            out_file.write('\n')
        out_file.close()
    print "Data split finished!"


def nbest2rerank(input_file, alpha):
    print "Start to turn Nbest format to rerank format for file: ", input_file
    in_lines = open(input_file,'rU').readlines()
    
    sentence_list = []
    goldlabel_list = []
    predictlabel_list = []
    sentence = []
    goldlabel = []
    predictlabel = []
    unique_sentence_count = 0
    probability = 0.0
    probability_list = []
    for line in in_lines:
        if line[0] == '#':
            if '# 0' in line:
                unique_sentence_count += 1
            nbest_info_pair = line.strip('\n').split(' ')
            probability = nbest_info_pair[2]
            continue
        elif len(line) < 2:
            sentence_list.append(sentence)
            goldlabel_list.append(goldlabel)
            predictlabel_list.append(predictlabel)
            probability_list.append(probability)
            sentence = []
            goldlabel = []
            predictlabel = []
            probability = 0.0
            continue
        else:
            pairs = line.strip('\n').split(' ')
            sentence.append(pairs[0])
            goldlabel.append(pairs[2])
            predictlabel.append(pairs[3])
    # count sentence num and unique sentence num

    sentence_num = len(sentence_list)
    out_file = open(input_file+'.train','w')
    right_count = 0
    line_number = 0
    the_sentence = []
    new_sentence_count = 0
    train2test_flag = True  ## make sure the train2test file just happens once
    gold_exist = False
    add_best = False
    for idx in range(0, sentence_num):
        ## If sentence not the same with previous, insert #*#NewSentence with sentence number
        if sentence_list[idx] != the_sentence:
            new_sentence_count += 1
            the_sentence = sentence_list[idx]

            # add best sentence if gold not exist in training data
            if (add_best)&(idx>0)&(gold_exist == False):
                for idy in range(0,len(sentence_list[idx-1])):
                    if goldlabel_list[idx-1][idy] == 'O':
                        out_file.write(sentence_list[idx-1][idy]+ ' [D]' + sentence_list[idx-1][idy]+ ' [G]' + goldlabel_list[idx-1][idy] + ' [P]' + goldlabel_list[idx-1][idy] + ' [T]' + float2class(probability_list[idx])+ ' [R]' + probability_list[idx-1] + ' True\n')
                    else:
                        # replace word with predict entities if sentence longer than 3
                        if len(sentence_list[idx-1]) > 0:
                            out_file.write(goldlabel_list[idx-1][idy]+ ' [D]' + sentence_list[idx-1][idy]+ ' [G]'+ goldlabel_list[idx-1][idy] + ' [P]' + goldlabel_list[idx-1][idy] + ' [T]' + float2class(probability_list[idx])+ ' [R]' + probability_list[idx-1] + ' True\n')
                        else:
                            out_file.write(sentence_list[idx-1][idy]+ ' [D]' + sentence_list[idx-1][idy]+ ' [G]' + goldlabel_list[idx-1][idy] + ' [P]' + goldlabel_list[idx-1][idy] + ' [T]' + float2class(probability_list[idx])+ ' [R]' + probability_list[idx-1] + ' True\n')
                out_file.write('\n')
            gold_exist = False
            ## split data into train and test files
            if (new_sentence_count > alpha*unique_sentence_count)&train2test_flag:
                out_file.close()
                out_file = open(input_file+'.test','w')
                train2test_flag =False
                add_best = False
            # out_file.write("#*#NewSentence:" + str(new_sentence_count) +"\n\n")
        if goldlabel_list[idx] == predictlabel_list[idx]:
            for idy in range(0,len(sentence_list[idx])):
                if predictlabel_list[idx][idy] == 'O':
                    out_file.write(sentence_list[idx][idy]+ ' [D]' + sentence_list[idx][idy]+ ' [G]' + goldlabel_list[idx][idy] + ' [P]' + predictlabel_list[idx][idy] + ' [T]' + float2class(probability_list[idx])+ ' [R]' + probability_list[idx] + ' True\n')
                else:
                    # replace word with predict entities if sentence longer than 3
                    if len(sentence_list[idx]) > 0:
                        out_file.write(predictlabel_list[idx][idy]+ ' [D]' + sentence_list[idx][idy]+ ' [G]'+ goldlabel_list[idx][idy] + ' [P]' + predictlabel_list[idx][idy] + ' [T]' + float2class(probability_list[idx])+ ' [R]' + probability_list[idx] + ' True\n')
                    else:
                        out_file.write(sentence_list[idx][idy]+ ' [D]' + sentence_list[idx][idy]+ ' [G]' + goldlabel_list[idx][idy] + ' [P]' + predictlabel_list[idx][idy] + ' [T]' + float2class(probability_list[idx])+ ' [R]' + probability_list[idx] + ' True\n')
            right_count += 1
            gold_exist = True
        else:
            for idy in range(0,len(sentence_list[idx])):
                if predictlabel_list[idx][idy] == 'O':
                    out_file.write(sentence_list[idx][idy]+ ' [D]'+ sentence_list[idx][idy]+ ' [G]' + goldlabel_list[idx][idy] + ' [P]' + predictlabel_list[idx][idy] + ' [T]' + float2class(probability_list[idx])+ ' [R]' + probability_list[idx] + ' False\n')
                else:
                    # replace word with predict entities
                    if len(sentence_list[idx]) > 0:
                        out_file.write(predictlabel_list[idx][idy]+ ' [D]' + sentence_list[idx][idy]+ ' [G]' + goldlabel_list[idx][idy] + ' [P]' + predictlabel_list[idx][idy] + ' [T]' + float2class(probability_list[idx])+ ' [R]' + probability_list[idx] + ' False\n')
                    else:
                        out_file.write(sentence_list[idx][idy]+ ' [D]'+ sentence_list[idx][idy]+ ' [G]' + goldlabel_list[idx][idy] + ' [P]' + predictlabel_list[idx][idy] + ' [T]' + float2class(probability_list[idx])+ ' [R]' + probability_list[idx] + ' False\n')
        out_file.write('\n')
             

    out_file.close()
    print "Nbest2rerank finished! Full sentence number: ", sentence_num
    print "BestIncludeNum: ", right_count, "; Unique sentence Num: ", unique_sentence_count, "; Rate: ", (right_count+0.0)/unique_sentence_count



def nbest2rerank_collapseLabel(input_file, alpha, add_best):
    print "Start to turn Nbest format to rerank format for file: ", input_file
    in_lines = open(input_file,'rU').readlines()
    sentence_list = []
    goldlabel_list = []
    predictlabel_list = []
    sentence = []
    goldlabel = []
    predictlabel = []
    unique_sentence_count = 0
    probability = 0.0
    probability_list = []
    for line in in_lines:
        if line[0] == '#':
            if '# 0' in line:
                unique_sentence_count += 1
            nbest_info_pair = line.strip('\n').split(' ')
            probability = nbest_info_pair[2]
            continue
        elif len(line) < 2:
            sentence_list.append(sentence)
            goldlabel_list.append(goldlabel)
            predictlabel_list.append(predictlabel)
            probability_list.append(probability)
            sentence = []
            goldlabel = []
            predictlabel = []
            probability = 0.0
            continue
        else:
            pairs = line.strip('\n').split(' ')
            sentence.append(pairs[0])
            goldlabel.append(pairs[2])
            predictlabel.append(pairs[3])
    # count sentence num and unique sentence num

    sentence_num = len(sentence_list)
    out_file = open('./b/'+input_file+'.txt','w')
    right_count = 0
    line_number = 0
    the_sentence = []
    new_sentence_count = 0
    gold_exist = False
    # add_best = False
    sentence_candidate_list = []
    sentences_list = []
    for idx in range(0, sentence_num):
        if (idx == sentence_num-1)|(sentence_list[idx] != the_sentence)&(idx > 0):
            if add_best:
                for idy in range(0, len(sentences_list)):
                    if sentences_list[idy][2] == sentences_list[idy][1]:
                        gold_exist = True
                        writeCollapseSentence(out_file, sentences_list[idy][0], sentences_list[idy][1], sentences_list[idy][2],sentences_list[idy][3], "True")
                    else:
                        writeCollapseSentence(out_file, sentences_list[idy][0], sentences_list[idy][1], sentences_list[idy][2],sentences_list[idy][3], "False")
                    if idy == len(sentences_list)-1:
                        if gold_exist == False:
                            print "Not find golden candidate, insert golden candidate... "
                            writeCollapseSentence(out_file, sentences_list[idy][0], sentences_list[idy][2], sentences_list[idy][2],sentences_list[idy][3], "True")
                        gold_exist = False
            else:
                highest_value = -1
                highest_position = -1
                for idy in range(0, len(sentences_list)):
                    a = []
                    b = []
                    c = []
                    d = []
                    value = -1
                    a,b,c,d,value = collapseEachSentence(sentences_list[idy][0], sentences_list[idy][1], sentences_list[idy][2])
                    if value > highest_value:
                        highest_value = value
                        highest_position = idy
                for idy in range(0, len(sentences_list)):
                    if idy == highest_position:
                        writeCollapseSentence(out_file, sentences_list[idy][0], sentences_list[idy][1], sentences_list[idy][2],sentences_list[idy][3], "True")
                    else:
                        writeCollapseSentence(out_file, sentences_list[idy][0], sentences_list[idy][1], sentences_list[idy][2],sentences_list[idy][3], "False")
            the_sentence = sentence_list[idx]            
            sentences_list = []
        sentence_candidate_list =[]
        sentence_candidate_list.append(sentence_list[idx])
        sentence_candidate_list.append(predictlabel_list[idx])
        sentence_candidate_list.append(goldlabel_list[idx])
        sentence_candidate_list.append(probability_list[idx])
        sentences_list.append(sentence_candidate_list)
        the_sentence = sentence_list[idx] 
        

        #### collect the last sentence
        if idx == sentence_num-2:
            sentence_candidate_list =[]
            sentence_candidate_list.append(sentence_list[sentence_num-1])
            sentence_candidate_list.append(predictlabel_list[sentence_num-1])
            sentence_candidate_list.append(goldlabel_list[sentence_num-1])
            sentence_candidate_list.append(probability_list[sentence_num-1])
            sentences_list.append(sentence_candidate_list)
            the_sentence = sentence_list[idx]
    out_file.close()



def collapseEachSentence(single_sentence_list, predict_label_list, gold_label_list):
    if (len(single_sentence_list) != len(predict_label_list))|(len(single_sentence_list) != len(gold_label_list)):
        print "Error in collapseEachSentence! Sentence, label and gold size mismatch!"
        return
    temp_label = ''
    catch_word = ''
    catch_gold = ''
    catch_predict = ''
    collapseSentence = []
    collapseOrigin = []
    collapseGold = []
    collapsePredict = []
    matched_label_size = 0
    for idx in range(0, len(gold_label_list)):
        if predict_label_list[idx] == gold_label_list[idx]:
            matched_label_size += 1
    precision_value = (matched_label_size+0.0)/len(gold_label_list)
    # print "length: ", len(gold_label_list), " precision: ", matched_label_size,"/", len(gold_label_list),'=',precision_value
    for idx in range(0, len(predict_label_list)):
        if "B-" in predict_label_list[idx]:
            if catch_word != '':
                collapseSentence.append(temp_label)
                collapseOrigin.append(catch_word)
                collapseGold.append(catch_gold)
                collapsePredict.append(catch_predict)
            catch_word = single_sentence_list[idx]
            temp_label = predict_label_list[idx].strip('B-')
            catch_gold = gold_label_list[idx]
            catch_predict = predict_label_list[idx]
        elif "I-" in predict_label_list[idx]:
            if temp_label == predict_label_list[idx].strip('I-'):
                catch_word += "*#*" + single_sentence_list[idx]
                catch_gold += "*#*" + gold_label_list[idx]
                catch_predict += "*#*" + predict_label_list[idx]
            else:
                if catch_word != '':
                    collapseSentence.append(temp_label)
                    collapseOrigin.append(catch_word)
                    collapseGold.append(catch_gold)
                    collapsePredict.append(catch_predict)
                collapseSentence.append(predict_label_list[idx])
                collapseOrigin.append(single_sentence_list[idx])
                collapseGold.append(gold_label_list[idx])
                collapsePredict.append(predict_label_list[idx])
                catch_word = ''
                temp_label = ''
                catch_gold = ''
                catch_predict = ''
        else:
            if catch_word != '':
                collapseSentence.append(temp_label)
                collapseOrigin.append(catch_word)
                collapseGold.append(catch_gold)
                collapsePredict.append(catch_predict)
            collapseSentence.append(single_sentence_list[idx])
            collapseOrigin.append(single_sentence_list[idx])
            collapseGold.append(gold_label_list[idx])
            collapsePredict.append(predict_label_list[idx])
            catch_word = ''
            temp_label = ''
            catch_gold = ''
            catch_predict = ''
        if (idx == len(predict_label_list) -1)&(catch_word != ''):
            collapseSentence.append(temp_label)
            collapseOrigin.append(catch_word)
            collapseGold.append(catch_gold)
            collapsePredict.append(catch_predict)

    if ((len(collapseSentence)!=len(collapseOrigin))|(len(collapseSentence)!=len(collapseGold))|(len(collapseSentence)!=len(collapsePredict))):
        print "Error in collapseEachSentence! Generate different size lists!"
        return
    return collapseSentence, collapseOrigin, collapseGold, collapsePredict, precision_value


def writeCollapseSentence(out_file, single_sentence_list, the_label_list, gold_label_list, probability, judge):
    collapseSentence, collapseOrigin, collapseGold, collapsePredict, precision_value = collapseEachSentence(single_sentence_list, the_label_list, gold_label_list)
    for idx in range(0,len(collapseSentence)):
        out_file.write(collapseSentence[idx] + ' [D]'+collapseOrigin[idx] + ' [G]' + collapseGold[idx] + ' [P]' + collapsePredict[idx] +' [V]'+ str(precision_value)+ ' [T]' + float2class(probability)+ ' [R]' + probability + ' ' +judge +'\n')
        # out_file.write(collapseSentence[idx] + ' [D]'+collapseOrigin[idx] + ' [G]' + collapseGold[idx] + ' [P]' + collapsePredict[idx] + ' [T]' + float2class(probability)+ ' [R]' + probability + ' ' +judge +'\n')
    out_file.write('\n')


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



## analysis the true distribution based on baseline predictor accuracy
def finalDataAnalysis(input_final_file, base_accuracy):
    print "probability threshold: ", base_accuracy
    lines = open(input_final_file,'rU').readlines()
    larger_accuracy_and_true = 0
    total_true = 0
    get_first = True
    collect_list = []
    for line in lines:
        if len(line) < 2:
            get_first = True
        else:
            if get_first:
                if float(line.split(' ')[5].strip('[R]'))>base_accuracy:    
                    collect_list.append(line)          
                if line.split(' ')[6]=="True\n":
                    total_true += 1
                get_first = False
    
    for each_line in collect_list:
        accuracy =  float(each_line.split(' ')[5].strip('[R]'))
        result = each_line.split(' ')[6].strip('\n')
        if result == "True":
            larger_accuracy_and_true += 1
    larger_rate = (len(collect_list)+0.0)/total_true
    true_larger_rate = (larger_accuracy_and_true+0.0)/len(collect_list)
    print "full_larger/total_true ", len(collect_list), '/', total_true,'=', larger_rate
    print "large_and_true/full_larger ", larger_accuracy_and_true, '/', len(collect_list),'=', true_larger_rate
    return larger_rate, true_larger_rate


if __name__ == '__main__':

    test_file = "conll03.test"
    train_file = "conll03.train"
    dev_file = "conll03.dev"

    # if sys.argv[1] == '-s':
    #     split_data(sys.argv[2], sys.argv[3])
    # elif sys.argv[1] == '-c':
    #     convert2conll(sys.argv[2])
    # elif sys.argv[1] == '-r':
    #     nbest2rerank(sys.argv[2], sys.argv[3])
    # elif sys.argv[1] == '-rca':
    #     nbest2rerank_collapseLabel(sys.argv[2], sys.argv[3], True)
    # elif sys.argv[1] == '-rcn':
    #     nbest2rerank_collapseLabel(sys.argv[2], sys.argv[3], False)
    # else:
    #     print "Data_clean do nothing!"

    # print "Finish data_clean/convert!"
    finalDataAnalysis('a/rrConll03.test.txt', 0.8)
    