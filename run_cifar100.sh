THEANO_FLAGS='floatX=float32' python cifar100.py --num_epochs 1000 --batch_size 100 \
 --learning_rate_cnn 0.001 --learning_rate_dense 0.1 --decay_rate 0.005 \
 --momentum0 0.5 --momentum1 0.95 --momentum_type normal \
 --delta 0.0 --regular none --gamma 1e-3 --patience 5
