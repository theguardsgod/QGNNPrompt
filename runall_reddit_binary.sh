DATA_DIR="/data/cyliu/dataset/reddit"
RUN_DIR="./run"
export PYTHONPATH="${PWD}:${PYTHONPATH}"
saves="GIN_int4_DQ_prompt"
# echo "int8 DQ"
# python reddit_binary/main.py --int8 --gc_per --lr 0.005 --DQ --low 0.0 --change 0.1 --wd 0.0002 --outdir ${RUN_DIR} --path ${DATA_DIR} | tee int8_dq.txt
# echo "int8 normal"
# python reddit_binary/main.py --int8 --ste_mom --lr 0.005 --wd 0.0002 --epochs 200 --outdir ${RUN_DIR} --path ${DATA_DIR} | tee int8.txt
echo "int4 DQ"
python reddit_binary/main.py --int4 --gc_per --lr 0.001 --DQ --low 0.1 --change 0.1 --wd 4e-5 --epochs 200 --outdir ${RUN_DIR} --path ${DATA_DIR} | tee ./runs/fp32_dq_p.txt
# echo "int4 normal"
# python reddit_binary/main.py --int4 --ste_mom --lr 0.05 --epochs 200 --outdir ${RUN_DIR} --path ${DATA_DIR} | tee int4.txt

# echo "fp32 normal"
# python reddit_binary/main.py --fp32 --ste_mom --lr 0.05 --epochs 200 --outdir ${RUN_DIR} --path ${DATA_DIR} --saves ${saves}  | tee fp32.txt
