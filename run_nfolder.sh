# cd tmp/
# rm *
# cd ..
THEANO_FLAGS='floatX=float32' python nbest_bi_lstm_cnn_crf.py --fine_tune --embedding senna --oov embedding --update momentum \
 --batch_size 10 --num_units 200 --num_filters 30 --learning_rate 0.015 --decay_rate 0.05 --grad_clipping 5 --regular none --dropout \
--output_prediction  --train "data/train.bioes-0" --dev "data/train.bioes0" --test "data/train.bioes0" \
 --embedding_dict "/home/yangjie/0.corpus/glove.6B.100d.txt" --patience 5 --nbest 10

# THEANO_FLAGS='floatX=float32' python nbest_bi_lstm_cnn_crf.py --fine_tune --embedding senna --oov embedding --update momentum \
#  --batch_size 10 --num_units 200 --num_filters 30 --learning_rate 0.015 --decay_rate 0.05 --grad_clipping 5 --regular none --dropout \
# --output_prediction  --train "data/train.bioes-1" --dev "data/train.bioes1" --test "data/train.bioes1" \
#  --embedding_dict "/home/yangjie/0.corpus/glove.6B.100d.txt" --patience 5 --nbest 10

#  THEANO_FLAGS='floatX=float32' python nbest_bi_lstm_cnn_crf.py --fine_tune --embedding senna --oov embedding --update momentum \
#  --batch_size 10 --num_units 200 --num_filters 30 --learning_rate 0.015 --decay_rate 0.05 --grad_clipping 5 --regular none --dropout \
# --output_prediction  --train "data/train.bioes-2" --dev "data/train.bioes2" --test "data/train.bioes2" \
#  --embedding_dict "/home/yangjie/0.corpus/glove.6B.100d.txt" --patience 5 --nbest 10

#  THEANO_FLAGS='floatX=float32' python nbest_bi_lstm_cnn_crf.py --fine_tune --embedding senna --oov embedding --update momentum \
#  --batch_size 10 --num_units 200 --num_filters 30 --learning_rate 0.015 --decay_rate 0.05 --grad_clipping 5 --regular none --dropout \
# --output_prediction  --train "data/train.bioes-3" --dev "data/train.bioes3" --test "data/train.bioes3" \
#  --embedding_dict "/home/yangjie/0.corpus/glove.6B.100d.txt" --patience 5 --nbest 10

#  THEANO_FLAGS='floatX=float32' python nbest_bi_lstm_cnn_crf.py --fine_tune --embedding senna --oov embedding --update momentum \
#  --batch_size 10 --num_units 200 --num_filters 30 --learning_rate 0.015 --decay_rate 0.05 --grad_clipping 5 --regular none --dropout \
# --output_prediction  --train "data/train.bioes-4" --dev "data/train.bioes4" --test "data/train.bioes4" \
#  --embedding_dict "/home/yangjie/0.corpus/glove.6B.100d.txt" --patience 5 --nbest 10