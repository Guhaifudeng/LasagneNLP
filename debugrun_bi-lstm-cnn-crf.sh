cd tmp/
rm *
cd ..
THEANO_FLAGS='floatX=float32' python debugbi_lstm_cnn_crf.py --fine_tune --embedding senna --oov embedding --update momentum \
 --batch_size 1 --num_units 200 --num_filters 30 --learning_rate 0.015 --decay_rate 0.05 --grad_clipping 5 --regular none --dropout \
--output_prediction  --train "data/debug.train.bioes" --dev "data/debug.dev.bioes" --test "data/debug.test.bioes" \
 --embedding_dict "data/debug.emb" --patience 5
