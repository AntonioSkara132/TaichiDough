# Preprocess rosbag

python3 scripts/preprocess_deformpath_episode.py \
    --bag-dir /home/antonio/diplomski_antonio/diplomski/data/deformpath_training/DeformPath2/<episode18_bag> \
    --output-dir /home/antonio/diplomski_antonio/diplomski/data/deformpath_training/preprocessed_dataset/episode18_kugla \
    --floor-y 0.0368 \
    --density 1200 \
    --num-particles 24000 \
    --max-raw-points 0 \
    --max-points 1024 \
    --frame 0 \
    --seed 0

# Run validation

# Run calibratio

python3 -m experiments.differentiable_mpm.calibrate fit \
    --config /home/antonio/diplomski_antonio/diplomski/data/deformpath_training/preprocessed_dataset/episode18_kugla/differentiable_mpm_smoke.json \
    --reference-policy frozen \
    --backend cpu \
    --precision f64 \
    --cpu-threads 1 \
    --end-frame 2 \
    --iterations 2 \
    --no-evaluate \
    --output-dir experiments/differentiable_mpm/runs/dynamics_fit_smoke --ignore-recompute-mismatch


<!-- python3 scripts/preprocess_deformpath_episode.py \ -->
<!--     --bag-dir /home/antonio/diplomski_antonio/diplomski/data/deformpath_training/DeformPath2/<episode18_bag> \ -->
<!--     --output-dir /home/antonio/diplomski_antonio/diplomski/data/deformpath_training/preprocessed_dataset/episode18_kugla \ -->
<!--     --floor-y 0.0368 \ -->
<!--     --density 1200 \ -->
<!--     --num-particles 24000 \ -->
<!--     --max-raw-points 0 \ -->
<!--     --max-points 1024 \ -->
<!--     --frame 0 \ -->
<!--     --seed 0 -->
