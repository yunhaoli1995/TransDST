CUDA_VISIBLE_DEVICES="0,1" python train.py \
                                    --data_root data/mwz2.1/ \
                                    --op_code '4' \
                                    --batch_size 32 \
                                    --grad_accumulation 1 \
                                    --dec_lr 1e-4\