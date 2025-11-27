pkill -9 -f "api.py"

nohup /home/cjj/anaconda3/envs/cosyvoice_vllm/bin/python api.py > logs/console.log  2>&1 &