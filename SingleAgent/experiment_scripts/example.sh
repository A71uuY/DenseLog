
# Specify --mode train for training.
# For SafeMotor, use Argument 'alg' to decide the algorithm.
# For Agricultural Spraying Drone, use Argument 'alg' and 'cons' to decide algorithm and task.
python ./examples/ddpg_farm_exp.py --mode train --constrained 1 --output ./output/exampleoutput_farm 