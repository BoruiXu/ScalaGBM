#!/bin/bash

num=1

./bin/scalagbm-train data=dataset/$1 objective=binary:logistic tree_method=hist n_gpus=$num n_trees=40 depth=6
#test
#./bin/scalagbm-predict data=dataset/$1.t objective=binary:logistic tree_method=hist n_gpus=$num 
