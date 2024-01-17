# For Agri Task1, use --cons disjunction.
# For Agri Task2, use --cons eq.
# Please remember to specify a correct output path!
python examples/ddpg_farm_exp.py --mode train --cons disjunction --alg pdl --constrained 1 --max_lag 20 --min_lag 2 --velo 0 --output output/

python ./examples/ddpg_farm_exp.py --mode test --constrained 1 --cons disjunction --output ./output/ --weights ./output/weights/ddpg_farm.h5f