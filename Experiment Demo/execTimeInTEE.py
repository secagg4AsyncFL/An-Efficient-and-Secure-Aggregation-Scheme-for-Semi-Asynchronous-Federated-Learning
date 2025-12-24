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
print(f"残差网络：")


# 记录开始时间
start_time = time.time()

# 加载所有的ResNet文件
resnet_files = [f"../helper-files/resnet/resnet{i}.npz" for i in range(1, 11)]

# 初始化一个字典来存储聚合后的数组
aggregated_results = {}

# 聚合每个数组（arr1到arr63）
for i in range(1, 60):
    arrays_to_aggregate = []
    # 加载每个文件并收集对应的数组
    for file in resnet_files:
        data = np.load(file)
        arrays_to_aggregate.append(data[f'arr{i}'])
    
    # 聚合并保存结果
    aggregated_results[f'arr{i}'] = agg_func(*arrays_to_aggregate)
# np.savez('resnet_aggr_10.npz', arrays_to_aggregate)

# 计算执行时间
end_time = time.time()
execution_time = end_time - start_time
print(f"模型*10 聚合时间: {execution_time} 秒")

# 关闭所有文件对象
for file in resnet_files:
    np.load(file).close()

# 手动删除文件变量并清理内存
del resnet_files
gc.collect()


# 记录开始时间
start_time = time.time()

# 加载所有的ResNet文件
resnet_files = [f"../helper-files/resnet/resnet{i}.npz" for i in range(1, 21)]

# 初始化一个字典来存储聚合后的数组
aggregated_results = {}

# 聚合每个数组（arr1到arr63）
for i in range(1, 60):
    arrays_to_aggregate = []
    # 加载每个文件并收集对应的数组
    for file in resnet_files:
        data = np.load(file)
        arrays_to_aggregate.append(data[f'arr{i}'])
    
    # 聚合并保存结果
    aggregated_results[f'arr{i}'] = agg_func(*arrays_to_aggregate)

# np.savez('resnet_aggr_20.npz', arrays_to_aggregate)

# 计算执行时间
end_time = time.time()
execution_time = end_time - start_time
print(f"模型*20 聚合时间: {execution_time:.6f} 秒")

# 关闭所有文件对象
for file in resnet_files:
    np.load(file).close()

# 手动删除文件变量并清理内存
del resnet_files
gc.collect()


# 记录开始时间
start_time = time.time()

# 加载所有的ResNet文件
resnet_files = [f"../helper-files/resnet/resnet{i}.npz" for i in range(1, 31)]

# 初始化一个字典来存储聚合后的数组
aggregated_results = {}

# 聚合每个数组（arr1到arr63）
for i in range(1, 60):
    arrays_to_aggregate = []
    # 加载每个文件并收集对应的数组
    for file in resnet_files:
        data = np.load(file)
        arrays_to_aggregate.append(data[f'arr{i}'])
    
    # 聚合并保存结果
    aggregated_results[f'arr{i}'] = agg_func(*arrays_to_aggregate)

# np.savez('resnet_aggr_30.npz', arrays_to_aggregate)

# 计算执行时间
end_time = time.time()
execution_time = end_time - start_time
print(f"模型*30 聚合时间: {execution_time:.6f} 秒")

# 关闭所有文件对象
for file in resnet_files:
    np.load(file).close()

# 手动删除文件变量并清理内存
del resnet_files
gc.collect()


# 记录开始时间
start_time = time.time()

# 加载所有的ResNet文件
resnet_files = [f"../helper-files/resnet/resnet{i}.npz" for i in range(1, 41)]

# 初始化一个字典来存储聚合后的数组
aggregated_results = {}

# 聚合每个数组（arr1到arr63）
for i in range(1, 60):
    arrays_to_aggregate = []
    # 加载每个文件并收集对应的数组
    for file in resnet_files:
        data = np.load(file)
        arrays_to_aggregate.append(data[f'arr{i}'])
    
    # 聚合并保存结果
    aggregated_results[f'arr{i}'] = agg_func(*arrays_to_aggregate)
# np.savez('resnet_aggr_40.npz', arrays_to_aggregate)

# 计算执行时间
end_time = time.time()
execution_time = end_time - start_time
print(f"模型*40 聚合时间: {execution_time:.6f} 秒")

# 关闭所有文件对象
for file in resnet_files:
    np.load(file).close()

# 手动删除文件变量并清理内存
del resnet_files
gc.collect()


# 记录开始时间
start_time = time.time()

# 加载所有的ResNet文件
resnet_files = [f"../helper-files/resnet/resnet{i}.npz" for i in range(1, 51)]

# 初始化一个字典来存储聚合后的数组
aggregated_results = {}

# 聚合每个数组（arr1到arr63）
for i in range(1, 60):
    arrays_to_aggregate = []
    # 加载每个文件并收集对应的数组
    for file in resnet_files:
        data = np.load(file)
        arrays_to_aggregate.append(data[f'arr{i}'])
    
    # 聚合并保存结果
    aggregated_results[f'arr{i}'] = agg_func(*arrays_to_aggregate)
    
# np.savez('resnet_aggr_50.npz', arrays_to_aggregate)

# 计算执行时间
end_time = time.time()
execution_time = end_time - start_time

print(f"模型*50 聚合时间: {execution_time:.6f} 秒")

# 关闭所有文件对象
for file in resnet_files:
    np.load(file).close()

# 手动删除文件变量并清理内存
del resnet_files
gc.collect()


# 记录开始时间
start_time = time.time()

# 加载所有的ResNet文件
resnet_files = [f"../helper-files/resnet/resnet{i}.npz" for i in range(1, 61)]

# 初始化一个字典来存储聚合后的数组
aggregated_results = {}

# 聚合每个数组（arr1到arr63）
for i in range(1, 60):
    arrays_to_aggregate = []
    # 加载每个文件并收集对应的数组
    for file in resnet_files:
        data = np.load(file)
        arrays_to_aggregate.append(data[f'arr{i}'])
    
    # 聚合并保存结果
    aggregated_results[f'arr{i}'] = agg_func(*arrays_to_aggregate)
# np.savez('resnet_aggr_60.npz', arrays_to_aggregate)

# 计算执行时间
end_time = time.time()
execution_time = end_time - start_time
print(f"模型*60 聚合时间: {execution_time:.6f} 秒")

# 关闭所有文件对象
for file in resnet_files:
    np.load(file).close()

# 手动删除文件变量并清理内存
del resnet_files
gc.collect()


print(f"残差网络优化：")

start_time = time.time()

random_float = random.random()
r1 = np.load("../helper-files/resnet/resnet1.npz")

modified_arrays = {}
# Loop through all arrays in the .npz file and add random_float
for key in r1.files:
    modified_arrays[key] = r1[key] + random_float

# 保存修改后的数据到新的 .npz 文件
# np.savez("./resnet_modified.npz", **modified_arrays)

end_time = time.time()
execution_time = end_time - start_time

print(f"优化后：{execution_time}")

np.load("../helper-files/resnet/resnet1.npz").close()
del r1
gc.collect()

########################################################################
###############################  C N N   ###############################
########################################################################

print(f"CNN：")
# 记录开始时间
start_time = time.time()

# 使用循环加载所有的 .npz 文件
files = [f"../helper-files/cnn/cnn{i}.npz" for i in range(1, 11)]
data_dict = {f"arr{i}": [] for i in range(1, 11)}  # 用来存储每个 'arr' 对应的数据

# 加载所有数据
for file in files:
    data.close()
    with np.load(file) as data:
        for i in range(1, 11):
            data_dict[f"arr{i}"].append(data[f"arr{i}"])
    data.close()
# 聚合数据并保存
aggregated_data = {f"arr{i}": agg_func(*data_dict[f"arr{i}"]) for i in range(1, 11)}

# 保存到一个新的 .npz 文件
# np.savez('cnn_aggr_10.npz', **aggregated_data)


end_time = time.time()
execution_time = end_time - start_time
print(f"模型*10：{execution_time}")


# 记录开始时间
start_time = time.time()

# 使用循环加载所有的 .npz 文件
files = [f"../helper-files/cnn/cnn{i}.npz" for i in range(1, 21)]
data_dict = {f"arr{i}": [] for i in range(1, 21)}  # 用来存储每个 'arr' 对应的数据

# 加载所有数据
for file in files:
    with np.load(file) as data:
        for i in range(1, 11):
            data_dict[f"arr{i}"].append(data[f"arr{i}"])

# 聚合数据并保存
aggregated_data = {f"arr{i}": agg_func(*data_dict[f"arr{i}"]) for i in range(1, 11)}

# 保存到一个新的 .npz 文件
# np.savez('cnn_aggr.npz_20', **aggregated_data)


end_time = time.time()
execution_time = end_time - start_time
print(f"模型*20：{execution_time}")


# 记录开始时间
start_time = time.time()

# 使用循环加载所有的 .npz 文件
files = [f"../helper-files/cnn/cnn{i}.npz" for i in range(1, 31)]
data_dict = {f"arr{i}": [] for i in range(1, 31)}  # 用来存储每个 'arr' 对应的数据

# 加载所有数据
for file in files:
    with np.load(file) as data:
        for i in range(1, 11):
            data_dict[f"arr{i}"].append(data[f"arr{i}"])

# 聚合数据并保存
aggregated_data = {f"arr{i}": agg_func(*data_dict[f"arr{i}"]) for i in range(1, 11)}

# 保存到一个新的 .npz 文件
# np.savez('cnn_aggr_30.npz', **aggregated_data)


end_time = time.time()
execution_time = end_time - start_time
print(f"模型*30：{execution_time}")


# 记录开始时间
start_time = time.time()

# 使用循环加载所有的 .npz 文件
files = [f"../helper-files/cnn/cnn{i}.npz" for i in range(1, 41)]
data_dict = {f"arr{i}": [] for i in range(1, 41)}  # 用来存储每个 'arr' 对应的数据

# 加载所有数据
for file in files:
    with np.load(file) as data:
        for i in range(1, 11):
            data_dict[f"arr{i}"].append(data[f"arr{i}"])

# 聚合数据并保存
aggregated_data = {f"arr{i}": agg_func(*data_dict[f"arr{i}"]) for i in range(1, 11)}

# 保存到一个新的 .npz 文件
# np.savez('cnn_aggr_40.npz', **aggregated_data)


end_time = time.time()
execution_time = end_time - start_time
print(f"模型*40：{execution_time}")


# 记录开始时间
start_time = time.time()

# 使用循环加载所有的 .npz 文件
files = [f"../helper-files/cnn/cnn{i}.npz" for i in range(1, 51)]
data_dict = {f"arr{i}": [] for i in range(1, 51)}  # 用来存储每个 'arr' 对应的数据

# 加载所有数据
for file in files:
    with np.load(file) as data:
        for i in range(1, 11):
            data_dict[f"arr{i}"].append(data[f"arr{i}"])

# 聚合数据并保存
aggregated_data = {f"arr{i}": agg_func(*data_dict[f"arr{i}"]) for i in range(1, 11)}

# 保存到一个新的 .npz 文件
# np.savez('cnn_aggr_50.npz', **aggregated_data)


end_time = time.time()
execution_time = end_time - start_time
print(f"模型*50：{execution_time}")


# 记录开始时间
start_time = time.time()

# 使用循环加载所有的 .npz 文件
files = [f"../helper-files/cnn/cnn{i}.npz" for i in range(1, 61)]
data_dict = {f"arr{i}": [] for i in range(1, 61)}  # 用来存储每个 'arr' 对应的数据

# 加载所有数据
for file in files:
    with np.load(file) as data:
        for i in range(1, 11):
            data_dict[f"arr{i}"].append(data[f"arr{i}"])

# 聚合数据并保存
aggregated_data = {f"arr{i}": agg_func(*data_dict[f"arr{i}"]) for i in range(1, 11)}

# 保存到一个新的 .npz 文件
# np.savez('cnn_aggr_60.npz', **aggregated_data)


end_time = time.time()
execution_time = end_time - start_time
print(f"模型*60：{execution_time}")


start_time = time.time()

random_float = random.random()
r1 = np.load("../helper-files/cnn/cnn1.npz")

modified_arrays = {}
# Loop through all arrays in the .npz file and add random_float
for key in r1.files:
    modified_arrays[key] = r1[key] + random_float

# 保存修改后的数据到新的 .npz 文件
# np.savez("./resnet_modified.npz", **modified_arrays)

end_time = time.time()
execution_time = end_time - start_time

print(f"优化后：{execution_time}")

np.load("./resnet/resnet1.npz").close()
del r1
gc.collect()