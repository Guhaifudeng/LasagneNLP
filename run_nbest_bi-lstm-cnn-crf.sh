cd tmp/
rm *
cd ..
THEANO_FLAGS='floatX=float32' python nbest_bi_lstm_cnn_crf.py --fine_tune --embedding senna --oov embedding --update momentum \
 --batch_size 10 --num_units 200 --num_filters 30 --learning_rate 0.015 --decay_rate 0.05 --grad_clipping 5 --regular none --dropout \
--output_prediction  --train "../NNNamedEntity/data/CONLL03NER/train.bioes" --dev "../NNNamedEntity/data/CONLL03NER/dev.bioes" --test "../NNNamedEntity/data/CONLL03NER/test.bioes" \
 --embedding_dict "/home/yangjie/Corpus/glove.6B.100d.txt" --patience 5 --nbest 10
