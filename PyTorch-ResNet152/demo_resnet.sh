export MASTER_ADDR=localhost
export MASTER_PORT=12355
/opt/amazon/openmpi/bin/mpirun -n 8 --bind-to core --map-by slot:PE=6 --rank-by core --report-bindings --allow-run-as-root \
    python3 train.py --model=resnet152 --device=hpu --batch-size=256 --epochs=90 --workers=10 \
    --dl-worker-type=MP --print-freq=10 --output-dir=. --seed=123 --hmp --hmp-bf16 ./ops_bf16_Resnet.txt \
    --hmp-fp32 ./ops_fp32_Resnet.txt --custom-lr-values 0.275 0.45 0.625 0.8 0.08 0.008 0.0008 \
    --custom-lr-milestones 1 2 3 4 30 60 80 --deterministic --dl-time-exclude=False