import sys
import random
from utils import *
from algo import *

data_name = sys.argv[1]
algorithm = sys.argv[2]
D = int(sys.argv[3])
num = int(sys.argv[4])
print(data_name, algorithm, D, num)
training_rating, test_rating, training_pos, test_pos= load_data("./tmp_dataset/ML-100K-{}".format(data_name))
print(len(training_pos), len(test_pos))
print(len(test_pos) / len(test_pos + training_pos), D)

training_X = deepcopy(training_rating)
test_X = deepcopy(test_rating)

if algorithm == "CP":
    paras = 0.005, 100, 0.01
else:
    paras = 0.003, 100, 0.01
    
result = []
for i in range(10):
    print("Count: {}".format(i))
    r0 = pure_sgd(paras, training_X, test_X, training_pos, test_pos, D, 
                  flag=algorithm, real_data=True)
    r1 = input_sgd(paras, training_X, test_X, training_pos,test_pos, D, 
                   flag=algorithm, real_data=True)
    r2 = gradient_sgd(paras, training_X, test_X, training_pos, test_pos, D, 
                      flag=algorithm, real_data=True) 
    r3 = output_sgd(paras, training_X, test_X, training_pos, test_pos, D, 
                    flag=algorithm, real_data=True)
#     clear_output()
    result.append([r0, r1, r2, r3])
    save_data(result, "./result/TC-DP/RD-S{}-{}-v{}-NO.{}".format(data_name, algorithm, num, i+1))
