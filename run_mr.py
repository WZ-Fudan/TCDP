
from easydl import clear_output
import random
from utils import *
from algo import *
import sys

N1 = 20
N2 = 20
N3 = 20

D = 3
mr = float(sys.argv[1])
flag = sys.argv[2]
if flag == "CP":
    TX, NX = generate_X_CP(N1, N2, N3, D, p=1, SNR=1)
elif flag == "Tucker":
  TX, NX = generate_X_Tucker(N1, N2, N3, D, p=1, SNR=1)
else:
  raise Exception("Invalid algorithm!!!")
result = []

for i in range(10):
    print("Count: {}, mr={}".format(i, mr))
    training_X, test_X, training_pos, test_pos = split_X(NX, TX, mr=mr, tr=0.8)
    r0 = pure_sgd((0.005, 100, 0.01), training_X, test_X, training_pos, 
                  test_pos, D, flag="Tucker")
    r1 = input_sgd((0.005, 100, 0.01), training_X, test_X, training_pos,
                  test_pos, D, flag="Tucker", real_data=False)
    r2 = gradient_sgd((0.005, 100, 0.01), training_X, test_X, training_pos,
                  test_pos, D, flag="Tucker", real_data=False)
    r3 = output_sgd((0.005, 100, 0.01), training_X, test_X, training_pos, 
                  test_pos, D, flag="Tucker", real_data=False)
    clear_output()
    result.append([r0, r1, r2, r3])
save_data(result, "./result/TC-DP/{}mr-{}tr-{}N3-{}D-{}-v1(c=1, n=10)".format(mr, '0.8', N3, D, flag))


