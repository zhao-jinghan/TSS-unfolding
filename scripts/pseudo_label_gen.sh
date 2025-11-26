# Prior to running this script, prepare the data according to the relevant config (see the data preparation in the appendix).

python datasets/build_CoP/build_CoP.py --cfg configs/build_CoP/graph_task.yml
python datasets/build_CoP/build_CoP.py --cfg configs/build_CoP/graph_step.yml
python datasets/build_CoP/build_CoP.py --cfg configs/build_CoP/graph_state.yml
