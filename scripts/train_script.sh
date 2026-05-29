#!/bin/bash

optimizer='adamdst'
blocksize=1
op_decay_steps=10

for size in "240m"; do
for steps in 20000; do
for lr in 2e-3; do
for density in 0.4; do
for bs in 128; do
for freq in 100; do
for update_r in 0.2; do

sbatch ./scripts/train_llm_time_mul_1_unit.sh $size $density $steps $bs $lr $update_r $freq $op_decay_steps $blocksize

done
done
done
done
done
done
done