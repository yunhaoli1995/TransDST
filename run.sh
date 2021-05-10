CUDA_VISIBLE_DEVICES="0,1,2,3" python TransDST_train.py \
                                    --data_root data/mwz2.1/ \
                                    --op_code '2' \
                                    --batch_size 32 \
                                    --grad_accumulation 1\
                                    --dec_lr 1e-4\
                                    --num_decoder_layers 2 \
                                    --model TransDST \
                                    --vocab_path assets/TransDST_vocabV2.txt \
                                    --n_epochs 10 \

# CUDA_VISIBLE_DEVICES="0,1,2,3" python TransDST_train.py \
#                                     --data_root data/mwz2.1/ \
#                                     --op_code '4' \
#                                     --batch_size 32 \
#                                     --grad_accumulation 1\
#                                     --dec_lr 1e-4\
#                                     --num_decoder_layers 2 \
#                                     --model TransDST \
#                                     --vocab_path assets/TransDST_vocabV2.txt \

# CUDA_VISIBLE_DEVICES="0,1,2,3" python TransDST_train.py \
#                                     --data_root data/mwz2.1/ \
#                                     --op_code '4' \
#                                     --batch_size 32 \
#                                     --grad_accumulation 1\
#                                     --dec_lr 1e-4\
#                                     --num_decoder_layers 2 \
#                                     --model TransDSTV3 \
#                                     --vocab_path assets/TransDST_vocabV2.txt \
#                                     --word_dropout 0 \
#                                     --not_shuffle_state \

# CUDA_VISIBLE_DEVICES="0,1,2,3" python TransDST_train.py \
#                                     --data_root data/mwz2.1/ \
#                                     --op_code '4' \
#                                     --batch_size 32 \
#                                     --grad_accumulation 1\
#                                     --dec_lr 1e-4\
#                                     --num_decoder_layers 2 \
#                                     --model TransDSTV3 \
#                                     --vocab_path assets/TransDST_vocabV2.txt \
#                                     --not_shuffle_state \
                                    