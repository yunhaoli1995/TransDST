CUDA_VISIBLE_DEVICES="4,5,6,7" python TransDST_train.py \
        --data_root data/mwz2.1/ \
        --op_code '2' \
        --batch_size 16 \
        --grad_accumulation 1\
        --dec_lr 1e-4\
        --num_decoder_layers 2 \
        --model TransDST \
        --word_dropout 0.1 \
        --vocab_path assets/TransDST_vocabV2.txt \
        --n_epochs 10 \
        --corrupt_method '' \
        --corrupt_p 0 \

CUDA_VISIBLE_DEVICES="4,5,6,7" python TransDST_train.py \
        --data_root data/mwz2.1/ \
        --op_code '2' \
        --batch_size 24 \
        --grad_accumulation 1\
        --dec_lr 1e-4\
        --num_decoder_layers 2 \
        --model TransDST \
        --word_dropout 0.1 \
        --vocab_path assets/TransDST_vocabV2.txt \
        --n_epochs 10 \
        --corrupt_method '' \
        --corrupt_p 0 \

CUDA_VISIBLE_DEVICES="4,5,6,7" python TransDST_train.py \
        --data_root data/mwz2.1/ \
        --op_code '2' \
        --batch_size 12 \
        --grad_accumulation 1\
        --dec_lr 1e-4\
        --num_decoder_layers 2 \
        --model TransDST \
        --word_dropout 0.1 \
        --vocab_path assets/TransDST_vocabV2.txt \
        --n_epochs 10 \
        --corrupt_method '' \
        --corrupt_p 0 \

CUDA_VISIBLE_DEVICES="4,5,6,7" python TransDST_train.py \
        --data_root data/mwz2.1/ \
        --op_code '2' \
        --batch_size 8 \
        --grad_accumulation 1\
        --dec_lr 1e-4\
        --num_decoder_layers 2 \
        --model TransDST \
        --word_dropout 0.1 \
        --vocab_path assets/TransDST_vocabV2.txt \
        --n_epochs 10 \
        --corrupt_method '' \
        --corrupt_p 0 \