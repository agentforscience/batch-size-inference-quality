cd /data/summereunann/neurico-workspaces/batch_size_inference_quality_20261023_120002_gh34ij56; source .venv/bin/activate
until grep -q "^holding 7.3" logs/gen_q3.log; do sleep 5; done; sleep 60
CONDS=b1,b1_rep B1_SUB=1 python src/generate.py unsloth/Llama-3.2-3B-Instruct l3b 7.5 bfloat16 >> logs/gen_l3b.log 2>&1
