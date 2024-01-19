CUDA_VISIBLE_DEVICES=0 python main.py --max_epochs=50  --num_workers=8 \
    --model_name_or_path roberta-large \
    --accumulate_grad_batches 4 \
    --batch_size 4 \
    --data_dir dataset/tacrev/k-shot/1-1 \
    --check_val_every_n_epoch 1 \
    --data_class WIKI80 \
    --max_seq_length 256 \
    --model_class RobertaForPrompt \
    --wandb \
    --litmodel_class RobertaLitModel \
    --task_name wiki80 \
    --lr 1e-5
