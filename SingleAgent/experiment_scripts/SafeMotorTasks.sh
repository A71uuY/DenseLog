# Use the following codes to perform SafeMotor-Task1 results.
# For SafeMotor-Task2, please replace cons_used 1 by cons_used 2.
# Please remember to specify a correct output path!
python examples/ddpg_motor.py --mode train --alg pdl --constrained 1 --cons_used 1 --max_lag 500 --min_lag 0 --beta 1 --output output/;

python ./examples/ddpg_motor.py --mode test --alg pdl --constrained 1 --cons_used 1 --output ./output --weights ./weights/ddpg_dcrl_series_weights.h5f


python examples/ddpg_motor.py --mode train --constrained 0 --cons_used 1 --output output/

python ./examples/ddpg_motor.py --mode test --constrained 0 --cons_used 1 --output ./output/ --weights ./output/weights/ddpg_dcrl_series_weights.h5f