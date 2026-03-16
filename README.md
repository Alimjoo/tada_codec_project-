python generate_data.py --limit 1000 --batch-size 4 

python train_codec.py --data arhip_program_ug_1000.pt --epochs 1 --batch-size 2 --device cuda

python test_codec.py --ckpt checkpoints/codec_epoch_0.pt --data arhip_program_ug_1000.pt --index 0 --device mps


python test_codec.py --ckpt checkpoints/codec_epoch_0.pt --data arhip_program_ug_1000.pt --index 0 --device cuda


