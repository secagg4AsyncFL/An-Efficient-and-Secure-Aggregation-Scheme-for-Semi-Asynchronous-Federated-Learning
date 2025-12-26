import numpy as np
import time
import random

import resource
import sys
import gc



def agg_func(*args):
    return np.sum(args, axis=0) / len(args)


########################################################################
###############################  ResNet  ###############################
########################################################################
print(f"ResNet:")


start_time = time.time()

resnet_files = [f"../helper-files/resnet/resnet{i}.npz" for i in range(1, 11)]

aggregated_results = {}

for i in range(1, 60):
    arrays_to_aggregate = []
    for file in resnet_files:
        data = np.load(file)
        arrays_to_aggregate.append(data[f'arr{i}'])
    
    aggregated_results[f'arr{i}'] = agg_func(*arrays_to_aggregate)
# np.savez('resnet_aggr_10.npz', arrays_to_aggregate)

end_time = time.time()
execution_time = end_time - start_time
print(f"模型*10 聚合时间: {execution_time} 秒")

for file in resnet_files:
    np.load(file).close()

del resnet_files
gc.collect()


start_time = time.time()

resnet_files = [f"../helper-files/resnet/resnet{i}.npz" for i in range(1, 21)]

aggregated_results = {}

for i in range(1, 60):
    arrays_to_aggregate = []
    for file in resnet_files:
        data = np.load(file)
        arrays_to_aggregate.append(data[f'arr{i}'])
    
    aggregated_results[f'arr{i}'] = agg_func(*arrays_to_aggregate)

# np.savez('resnet_aggr_20.npz', arrays_to_aggregate)

end_time = time.time()
execution_time = end_time - start_time
print(f"model*20 aggregation time: {execution_time:.6f} seconds")

# Close all file objects
for file in resnet_files:
    np.load(file).close()

# Manually delete file variables and clean up memory
del resnet_files
gc.collect()


start_time = time.time()
resnet_files = [f"../helper-files/resnet/resnet{i}.npz" for i in range(1, 31)]
aggregated_results = {}

for i in range(1, 60):
    arrays_to_aggregate = []
    for file in resnet_files:
        data = np.load(file)
        arrays_to_aggregate.append(data[f'arr{i}'])
    aggregated_results[f'arr{i}'] = agg_func(*arrays_to_aggregate)


end_time = time.time()
execution_time = end_time - start_time
print(f"model*30 aggregation time: {execution_time:.6f} seconds")

# Close all file objects
for file in resnet_files:
    np.load(file).close()
del resnet_files
gc.collect()

start_time = time.time()

resnet_files = [f"../helper-files/resnet/resnet{i}.npz" for i in range(1, 41)]

aggregated_results = {}

for i in range(1, 60):
    arrays_to_aggregate = []
    for file in resnet_files:
        data = np.load(file)
        arrays_to_aggregate.append(data[f'arr{i}'])
    
    aggregated_results[f'arr{i}'] = agg_func(*arrays_to_aggregate)
# np.savez('resnet_aggr_40.npz', arrays_to_aggregate)

end_time = time.time()
execution_time = end_time - start_time
print(f"model*40 aggregation time: {execution_time:.6f} seconds")

for file in resnet_files:
    np.load(file).close()
del resnet_files
gc.collect()



start_time = time.time()
resnet_files = [f"../helper-files/resnet/resnet{i}.npz" for i in range(1, 51)]

aggregated_results = {}

for i in range(1, 60):
    arrays_to_aggregate = []
    for file in resnet_files:
        data = np.load(file)
        arrays_to_aggregate.append(data[f'arr{i}'])
    
    # 聚合并保存结果
    aggregated_results[f'arr{i}'] = agg_func(*arrays_to_aggregate)
    
# np.savez('resnet_aggr_50.npz', arrays_to_aggregate)

end_time = time.time()
execution_time = end_time - start_time

print(f"model*50 aggregation time: {execution_time:.6f} seconds")

for file in resnet_files:
    np.load(file).close()

del resnet_files
gc.collect()


start_time = time.time()

resnet_files = [f"../helper-files/resnet/resnet{i}.npz" for i in range(1, 61)]
aggregated_results = {}

for i in range(1, 60):
    arrays_to_aggregate = []
    for file in resnet_files:
        data = np.load(file)
        arrays_to_aggregate.append(data[f'arr{i}'])
    
    aggregated_results[f'arr{i}'] = agg_func(*arrays_to_aggregate)
# np.savez('resnet_aggr_60.npz', arrays_to_aggregate)

end_time = time.time()
execution_time = end_time - start_time
print(f"model*60 aggregation time: {execution_time:.6f} seconds")

for file in resnet_files:
    np.load(file).close()

del resnet_files
gc.collect()


print(f"Residual Network Optimization:")

start_time = time.time()

random_float = random.random()
r1 = np.load("../helper-files/resnet/resnet1.npz")

modified_arrays = {}
# Loop through all arrays in the .npz file and add random_float
for key in r1.files:
    modified_arrays[key] = r1[key] + random_float

end_time = time.time()
execution_time = end_time - start_time

print(f"optimization time: {execution_time:.6f} seconds")

np.load("../helper-files/resnet/resnet1.npz").close()
del r1
gc.collect()

########################################################################
###############################  C N N   ###############################
########################################################################

print(f"CNN：")
start_time = time.time()

files = [f"../helper-files/cnn/cnn{i}.npz" for i in range(1, 11)]
data_dict = {f"arr{i}": [] for i in range(1, 11)}  

for file in files:
    data.close()
    with np.load(file) as data:
        for i in range(1, 11):
            data_dict[f"arr{i}"].append(data[f"arr{i}"])
    data.close()
aggregated_data = {f"arr{i}": agg_func(*data_dict[f"arr{i}"]) for i in range(1, 11)}




end_time = time.time()
execution_time = end_time - start_time
print(f"model*10 aggregation time: {execution_time:.6f} seconds")


start_time = time.time()

files = [f"../helper-files/cnn/cnn{i}.npz" for i in range(1, 21)]
data_dict = {f"arr{i}": [] for i in range(1, 21)}  

for file in files:
    with np.load(file) as data:
        for i in range(1, 11):
            data_dict[f"arr{i}"].append(data[f"arr{i}"])

aggregated_data = {f"arr{i}": agg_func(*data_dict[f"arr{i}"]) for i in range(1, 11)}



end_time = time.time()
execution_time = end_time - start_time
print(f"model*20 aggregation time: {execution_time:.6f} seconds")



start_time = time.time()

files = [f"../helper-files/cnn/cnn{i}.npz" for i in range(1, 31)]
data_dict = {f"arr{i}": [] for i in range(1, 31)}  


for file in files:
    with np.load(file) as data:
        for i in range(1, 11):
            data_dict[f"arr{i}"].append(data[f"arr{i}"])


aggregated_data = {f"arr{i}": agg_func(*data_dict[f"arr{i}"]) for i in range(1, 11)}




end_time = time.time()
execution_time = end_time - start_time
print(f"model*30 aggregation time: {execution_time:.6f} seconds")


start_time = time.time()

files = [f"../helper-files/cnn/cnn{i}.npz" for i in range(1, 41)]
data_dict = {f"arr{i}": [] for i in range(1, 41)}  

for file in files:
    with np.load(file) as data:
        for i in range(1, 11):
            data_dict[f"arr{i}"].append(data[f"arr{i}"])

aggregated_data = {f"arr{i}": agg_func(*data_dict[f"arr{i}"]) for i in range(1, 11)}



end_time = time.time()
execution_time = end_time - start_time
print(f"model*40 aggregation time: {execution_time:.6f} seconds")



start_time = time.time()

files = [f"../helper-files/cnn/cnn{i}.npz" for i in range(1, 51)]
data_dict = {f"arr{i}": [] for i in range(1, 51)}  
for file in files:
    with np.load(file) as data:
        for i in range(1, 11):
            data_dict[f"arr{i}"].append(data[f"arr{i}"])

aggregated_data = {f"arr{i}": agg_func(*data_dict[f"arr{i}"]) for i in range(1, 11)}


end_time = time.time()
execution_time = end_time - start_time
print(f"model*50 aggregation time: {execution_time:.6f} seconds")


start_time = time.time()


files = [f"../helper-files/cnn/cnn{i}.npz" for i in range(1, 61)]
data_dict = {f"arr{i}": [] for i in range(1, 61)}  


for file in files:
    with np.load(file) as data:
        for i in range(1, 11):
            data_dict[f"arr{i}"].append(data[f"arr{i}"])

aggregated_data = {f"arr{i}": agg_func(*data_dict[f"arr{i}"]) for i in range(1, 11)}



end_time = time.time()
execution_time = end_time - start_time
print(f"model*60 aggregation time: {execution_time:.6f} seconds")


start_time = time.time()

random_float = random.random()
r1 = np.load("../helper-files/cnn/cnn1.npz")

modified_arrays = {}
# Loop through all arrays in the .npz file and add random_float
for key in r1.files:
    modified_arrays[key] = r1[key] + random_float



end_time = time.time()
execution_time = end_time - start_time

print(f"model*1 optimization time: {execution_time:.6f} seconds")

np.load("./resnet/resnet1.npz").close()
del r1
gc.collect()