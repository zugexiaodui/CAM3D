python main.py ucf101 RGB \
     --arch resnet50 --num_segments 8 \
     --gd 20 --lr 0.002 --wd 1e-4 --lr_steps 40 80 --epochs 100 \
     --batch-size 12 -j 16 --dropout 0.5 --consensus_type=avg --eval-freq=1 \
     --shift --shift_div=8 --shift_place=blockres --npb --cca3d --suffix=rccax1_layer1_div4 \
     --resume checkpoint/TSM_ucf101_RGB_resnet50_shift8_blockres_avg_segment8_e100_cca3d_rccax1_layer1_div4/ckpt.pth.tar

     #--shift --shift_div=8 --shift_place=blockres --npb --cca3d --suffix=rccax1_layer2_div4 \
     #--resume checkpoint/TSM_ucf101_RGB_resnet50_shift8_blockres_avg_segment8_e100_cca3d_rccax1_layer2_div4/ckpt.pth.tar

     #--shift --shift_div=8 --shift_place=blockres --npb --cca3d --suffix=rccax1_r3_div16 \
     #--resume checkpoint/TSM_ucf101_RGB_resnet50_shift8_blockres_avg_segment8_e100_cca3d_rccax1_r3_div16/ckpt.pth.tar
