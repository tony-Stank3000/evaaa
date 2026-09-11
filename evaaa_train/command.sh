#! /bin/bash
# python train.py exp=dqn tag=tet seed=42 env.port=50000 env.time_scale=15 env.width=100 env.height=100 env.config=train-level-1.1-ScatteredResource env.screenRecordEnable=false env.dataRecordEnable=false
# python eval.py exp=dqn tag=tet seed=42 env.port=50000 env.time_scale=1 env.width=1000 env.height=1000 env.config=exp-damage env.screenRecordEnable=true env.dataRecordEnable=true checkpoint_path=/media/nas01/projects/Interoceptive-AI/evaaa_train/logs/runs/dqn/2025-05-14_12-48-13_dqn_tet/version_0/checkpoint/ckpt_1.ckpt
# python train.py exp=ppo tag=tet seed=42 env.port=50000 env.time_scale=1 env.width=1000 env.height=1000 env.config=exp-damage env.screenRecordEnable=true env.dataRecordEnable=true

python train.py exp=ppo_recurrent tag=ppor_level1_1 seed=42 env.port=50000 env.time_scale=15 env.width=100 env.height=100 env.config=train-level-1.1-Scattered-Cubes env.screenRecordEnable=false env.dataRecordEnable=false
