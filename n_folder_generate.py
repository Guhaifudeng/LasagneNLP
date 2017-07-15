# -*- coding: utf-8 -*-
# @Author: Jie
# @Date:   2017-02-16 17:33:29
# @Last Modified by:   Jie     @Contact: jieynlp@gmail.com
# @Last Modified time: 2017-02-20 16:49:10

import copy
import random


def n_folder_generate(input_file, n_folder):
	in_lines = open(input_file,'r').readlines()
	sentence_list = []
	sentence = ""
	for line in in_lines:
		if len(line) < 2:
			sentence_list.append(sentence)
			sentence = ""
		else:
			sentence += line
	sent_num = len(sentence_list)
	# random.shuffle(sentence_list)
	print "Read file:",input_file, ", sentence num:", sent_num
	blocks = []
	block_size = sent_num/n_folder+1
	for idx in range(n_folder):
		start = idx*block_size
		end = (idx+1)*block_size
		if end > sent_num:
			end = sent_num
		blocks.append(sentence_list[start:end])
	print len(blocks)
	new_sent_num = 0
	for idx in range(n_folder):
		new_blocks = copy.deepcopy(blocks)
		print len(new_blocks)
		dev_block = new_blocks[idx]
		del new_blocks[idx]
		train_blocks = new_blocks
		train_file = open(input_file+"-"+str(idx),'w')
		dev_file = open(input_file+str(idx),'w')
		for block in train_blocks:
			for sent in block:
				train_file.write(sent+"\n")
		for sent in dev_block:
			dev_file.write(sent+'\n')
		train_file.close()
		dev_file.close()





if __name__ == '__main__':
	# random.seed(1)
	n_folder_generate("data/train.bio",5)




