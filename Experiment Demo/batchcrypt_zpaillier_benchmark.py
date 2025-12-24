
from heu import numpy as hnp
from heu import phe
import numpy as np
from functools import wraps
import time
import json
from pathlib import Path
from datetime import datetime


# ============ 性能计时装饰器 ============

def timing_decorator(func):
    """函数执行时间装饰器，单位：毫秒"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = (end_time - start_time) * 1000  # 转为毫秒
        return result, execution_time
    return wrapper


# ============ 步骤 1: 密钥生成 ============

@timing_decorator
def key_generation(key_length=2048):
    """生成 ZPaillier 密钥"""
    kit = hnp.setup(phe.SchemaType.ZPaillier, key_length)
    encryptor = kit.encryptor()
    decryptor = kit.decryptor()
    evaluator = kit.evaluator()
    return (encryptor, decryptor, evaluator, kit)





# ============ 步骤 2: 梯度量化 ============

def quantize_matrix(matrix, bit_width=8, r_max=0.5):
    """
    梯度量化：将浮点梯度压缩为 8 位整数
    
    公式: q = sign(g) × ⌊|g| × (2^(bit_width-1) - 1) / r_max⌋
    
    参数:
        matrix: 浮点梯度矩阵
        bit_width: 量化位宽（默认 8 位）
        r_max: 动态范围最大值（默认 0.5）
    
    返回:
        量化后的浮点值（未舍入）
    """
    og_sign = np.sign(matrix)
    uns_matrix = matrix * og_sign  # 取绝对值
    uns_result = uns_matrix * (pow(2, bit_width - 1) - 1.0) / r_max
    result = og_sign * uns_result
    return result


# ============ 步骤 3: 随机舍入 ============

def stochastic_round(ori):
    """
    随机舍入：保持无偏性的概率舍入
    
    原理: 根据小数部分概率决定上取整或下取整
    好处: E[round(x)] = x，减少量化误差
    """
    rand = np.random.rand(len(ori))
    frac, _ = np.modf(ori)
    result = np.zeros(len(ori), dtype=np.int32)
    
    for i in range(len(ori)):
        if frac[i] >= 0:
            result[i] = np.floor(ori[i]) if frac[i] <= rand[i] else np.ceil(ori[i])
        else:
            result[i] = np.floor(ori[i]) if (-1 * frac[i]) > rand[i] else np.ceil(ori[i])
    
    return result.astype(np.int32)


def stochastic_round_matrix(ori):
    """
    矩阵随机舍入
    """
    _shape = ori.shape
    ori = ori.reshape(-1)
    result = stochastic_round(ori)
    result = result.reshape(_shape)
    return result


# ============ 步骤 4: 补码转换 ============

def true_to_two_comp(input_array, bit_width):
    """
    补码转换：有符号整数 → 无符号补码表示
    
    原理:
        正数: 保持不变
        负数: 2^(bit_width+1) + value
    
    便于后续批量打包
    """
    result = np.zeros(len(input_array), dtype=np.int32)
    for i in range(len(input_array)):
        if input_array[i] >= 0:
            result[i] = input_array[i]
        else:
            result[i] = 2 ** (bit_width + 1) + input_array[i]
    return result


# ============ 步骤 5: 批量打包 (核心优化) ============

def batch_pack(array, batch_size=16, bit_width=8, pad_zero=3):
    """
    批量打包：16 个梯度打包成 1 个大整数
    
    核心优化：将加密次数从 16 → 1，实现 16 倍加速
    
    流程:
        1. 每个补码占 (bit_width + pad_zero) 位
        2. 通过位移和拼接组合成大整数
        3. 示例 (bit_width=8, pad_zero=3):
           [a0, a1, ..., a15] → a15 << 165 | a14 << 154 | ... | a0
    
    返回:
        打包后的大整数数组
    """
    A_len = len(array)
    # 填充到 batch_size 的倍数
    if (A_len % batch_size) != 0:
        array = np.pad(array, (0, batch_size - (A_len % batch_size)), 'constant', constant_values=0)
    
    idx_range = int(len(array) / batch_size)
    idx_base = list(range(idx_range))
    batched_nums = np.zeros(idx_range, dtype=object)  # 使用 object 类型支持大整数
    
    # 批量打包
    for i in range(batch_size):
        idx_filter = [i + x * batch_size for x in idx_base]
        filtered_num = array[idx_filter]
        batched_nums = (batched_nums * pow(2, (bit_width + pad_zero))) + filtered_num
    
    return batched_nums, A_len


# ============ 步骤 2-6: 数据预处理（完整 BatchCrypt 流程）============

@timing_decorator
def preprocess_matrices(matrices, bit_width=8, r_max=0.5, batch_size=16, pad_zero=3):
    """
    数据预处理（BatchCrypt 完整流程）
    
    流程:
        1. 展平矩阵
        2. 梯度量化
        3. 随机舍入
        4. 补码转换
        5. 批量打包
    
    返回:
        processed: 打包后的大整数列表
        metadata: 元数据（原始形状、原始长度等）
    """
    processed = []
    metadata = []
    
    for matrix in matrices:
        # 步骤 1: 展平
        og_shape = matrix.shape
        flattened = matrix.reshape(-1)
        
        # 步骤 2: 量化
        quantized = quantize_matrix(flattened, bit_width, r_max)
        
        # 步骤 3: 随机舍入
        rounded = stochastic_round(quantized)
        
        # 步骤 4: 补码转换
        two_comp = true_to_two_comp(rounded, bit_width)
        
        # 步骤 5: 批量打包
        packed, original_len = batch_pack(two_comp, batch_size, bit_width, pad_zero)
        
        processed.append(packed)
        metadata.append({
            'shape': og_shape,
            'original_len': original_len,
            'batch_size': batch_size,
            'bit_width': bit_width,
            'pad_zero': pad_zero,
            'r_max': r_max
        })
        
        # FLOPs统计：量化、舍入、转换各算 1 次操作
        flops_counter.preprocess += 3 * original_len
    
    return processed, metadata


# ============ 步骤 6-7: 加密 (对打包后的大整数加密) ============

# ============ 步骤 6-7: 加密 (对打包后的大整数加密) ============

@timing_decorator
def encrypt_matrices(packed_matrices_list, metadata_list, encryptor, kit):
    """
    加密打包后的矩阵
    
    关键: 加密打包后的大整数（而非原始梯度）
    性能: 16 个梯度 → 1 次加密，实现 16 倍加速
    
    流程:
    1. 对每个打包后的大整数进行加密
    2. 使用 BigintEncoder 编码大整数
    3. 批量加密以提高效率
    """
    encrypted_matrices = []
    
    for packed_array, metadata in zip(packed_matrices_list, metadata_list):
        # 将所有打包值转换为Python int列表
        packed_ints = []
        for packed_value in packed_array:
            if isinstance(packed_value, np.ndarray):
                packed_ints.append(int(packed_value.item()))
            else:
                packed_ints.append(int(packed_value))
        
        # 批量编码和加密
        encoder = phe.BigintEncoder(phe.SchemaType.ZPaillier)
        heu_plaintext = kit.array(packed_ints, encoder)
        encrypted_array = encryptor.encrypt(heu_plaintext)
        
        encrypted_matrices.append(encrypted_array)
        
        # FLOPs统计：每次加密打包值相当于加密 batch_size 个元素
        flops_counter.encrypt += 2 * len(packed_ints)
    
    return encrypted_matrices


# ============ 步骤 8: 同态聚合 ============

# ============ 步骤 8: 同态聚合 ============

@timing_decorator
def homomorphic_aggregation(encrypted_matrices_list, evaluator):
    """
    同态聚合（加法）
    
    E(g1) + E(g2) = E(g1+g2)
    无需解密，直接在加密域运算
    
    注意: 操作的是打包后的密文，每次加法相当于聚合 16 个梯度
    
    参数:
        encrypted_matrices_list: 多个客户端的加密矩阵列表
        每个元素是一个 CiphertextArray，包含该客户端所有矩阵的加密打包值
    """
    # 初始化为第一个客户端的密文
    aggregated_matrices = []
    for enc_matrix in encrypted_matrices_list[0]:
        aggregated_matrices.append(enc_matrix)
    
    # 累加其他客户端的密文
    for client_encrypted_matrices in encrypted_matrices_list[1:]:
        for matrix_idx in range(len(aggregated_matrices)):
            # 直接对 CiphertextArray 进行同态加法
            aggregated_matrices[matrix_idx] = evaluator.add(
                aggregated_matrices[matrix_idx],
                client_encrypted_matrices[matrix_idx]
            )
            # FLOPs统计：每个矩阵的打包值数量
            # CiphertextArray 使用 .size 属性获取元素数量
            flops_counter.aggregate += aggregated_matrices[matrix_idx].size
    
    return aggregated_matrices


# ============ 步骤 9: 解包 ============

def two_comp_to_true(two_comp, bit_width=8, pad_zero=3):
    """
    补码转真值：处理溢出检测
    
    根据符号位判断:
        - 000: 正数
        - 001: 正溢出
        - 011: 负数
        - 110: 负溢出
    """
    if two_comp < 0:
        raise Exception("Error: not expecting negative value")
    
    sign = two_comp >> (bit_width - 1)
    literal = two_comp & (2 ** (bit_width - 1) - 1)
    
    if sign == 0:  # 正数 (0000)
        return literal
    elif sign == 4:  # 正数 (0100)
        return literal
    elif sign == 1:  # 正溢出 (0001)
        return pow(2, bit_width - 1) - 1
    elif sign == 3:  # 负数 (0011)
        return - 1 * (2 ** (bit_width - 1) - literal)
    elif sign == 7:  # 负数 (0111)
        return - 1 * (2 ** (bit_width - 1) - literal)
    elif sign == 6:  # 负溢出 (0110)
        print(f'  警告: 负溢出 {two_comp}')
        return - (pow(2, bit_width - 1) - 1)
    else:  # 未识别的溢出
        print(f'  警告: 未识别溢出 {two_comp}')
        return - (pow(2, bit_width - 1) - 1)


def unpack_batch(packed_array, original_len, batch_size=16, bit_width=8, pad_zero=3):
    """
    解包：从大整数提取 16 个补码
    
    逆向批量打包过程:
        1. 使用位掩码提取每个补码
        2. 转换补码为真值
        3. 截取到原始长度
    """
    num_ele_w_pad = batch_size * len(packed_array)
    un_batched_nums = np.zeros(num_ele_w_pad, dtype=np.int32)
    
    for i in range(batch_size):
        filter_mask = (pow(2, bit_width + pad_zero) - 1) << ((bit_width + pad_zero) * i)
        
        for j in range(len(packed_array)):
            packed_int = int(packed_array[j])  # 确保是 Python int
            two_comp = (filter_mask & packed_int) >> ((bit_width + pad_zero) * i)
            un_batched_nums[batch_size * j + batch_size - 1 - i] = two_comp_to_true(two_comp, bit_width, pad_zero)
    
    # 截取到原始长度（去除填充）
    un_batched_nums = un_batched_nums[:original_len]
    return un_batched_nums


# ============ 步骤 10: 反量化 ============

def unquantize_matrix(matrix, bit_width=8, r_max=0.5):
    """
    反量化：整数 → 浮点梯度
    
    公式: g = sign(q) × (|q| × r_max / (2^(bit_width-1) - 1))
    """
    matrix = matrix.astype(np.int32)
    og_sign = np.sign(matrix)
    uns_matrix = matrix * og_sign
    uns_result = uns_matrix * r_max / (pow(2, bit_width - 1) - 1.0)
    result = og_sign * uns_result
    return result.astype(np.float32)


# ============ 步骤 8-10: 解密与恢复 ============

@timing_decorator
def decrypt_matrices(encrypted_matrices, metadata_list, decryptor):
    """
    解密并恢复矩阵（完整 BatchCrypt 逆向流程）
    
    流程:
        1. 解密大整数
        2. 解包（提取补码）
        3. 反量化（恢复浮点值）
        4. 恢复原始形状
    """
    decrypted_matrices = []
    
    for encrypted_array, metadata in zip(encrypted_matrices, metadata_list):
        # 步骤 1: 批量解密打包值
        try:
            # 解密整个 CiphertextArray
            dec_array = decryptor.decrypt(encrypted_array)
            # PlaintextArray 使用索引访问每个元素
            arr_size = encrypted_array.size
            decrypted_packed = [int(dec_array[i]) for i in range(arr_size)]
            flops_counter.decrypt += 2 * len(decrypted_packed)
        except Exception as e:
            print(f"  解密出错: {e}")
            # 使用 .size 属性获取 CiphertextArray 的元素数量
            arr_size = encrypted_array.size
            decrypted_packed = [0] * arr_size
            flops_counter.decrypt += 2 * arr_size
        
        # 步骤 2: 解包
        unpacked = unpack_batch(
            decrypted_packed,
            metadata['original_len'],
            metadata['batch_size'],
            metadata['bit_width'],
            metadata['pad_zero']
        )
        
        # 步骤 3: 反量化
        unquantized = unquantize_matrix(
            unpacked,
            metadata['bit_width'],
            metadata['r_max']
        )
        
        # 步骤 4: 恢复形状
        restored = unquantized.reshape(metadata['shape'])
        decrypted_matrices.append(restored)
        
        # FLOPs统计：解包和反量化
        flops_counter.postprocess += 2 * metadata['original_len']
    
    return decrypted_matrices


# ============ 数据加载 ============

def load_matrices_from_npz(file_path):
    """从 npz 文件加载矩阵"""
    with np.load(file_path) as data:
        matrices = [data[key] for key in sorted(data.files)]
    return matrices


# ============ 主测试函数 ============

def run_benchmark(file_path1='./cnn/cnn.npz', file_path2='./cnn/cnn1.npz', key_length=2048):
    """
    运行完整的 BatchCrypt 性能测试
    
    测量以下阶段的耗时:
    1. 密钥生成
    2. 数据预处理 (量化 → 随机舍入 → 补码转换 → 批量打包)
    3. 加密 (对打包后的大整数加密)
    4. 同态聚合
    5. 解密 + 解包 + 反量化
    
    BatchCrypt 核心优化:
    - 梯度量化: 32位浮点 → 8位整数 (4x压缩)
    - 批量打包: 16个梯度 → 1个密文 (16x加速)
    - 总体提升: 理论加速比 16x，精度损失 < 1%
    """
    
    # 重置 FLOPs 计数器
    global flops_counter
    flops_counter = FlopsCounter()
    
    print("\n" + "="*80)
    print("BatchCrypt 完整流程性能测试 (量化+打包 + ZPaillier)")
    print("="*80)
    
    # 检查文件是否存在
    if not Path(file_path1).exists() or not Path(file_path2).exists():
        print(f"数据文件不存在: {file_path1} 或 {file_path2}")
        return None
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'key_length': key_length,
        'timings': {}
    }
    
    # ============ 步骤 1: 密钥生成 ============
    print(f"\n密钥生成 (key_length={key_length})")
    (encryptor, decryptor, evaluator, kit), keygen_time = key_generation(key_length)
    print(f"耗时: {keygen_time:.2f} ms")
    results['timings']['KeyGen'] = keygen_time
    
    # ============ 加载数据 ============
    print(f"\n加载数据")
    matrices1 = load_matrices_from_npz(file_path1)
    matrices2 = load_matrices_from_npz(file_path2)
    print(f"客户端 1: {len(matrices1)} 个矩阵")
    print(f"客户端 2: {len(matrices2)} 个矩阵")
    
    # ============ 步骤 2-6: 数据预处理 (量化+打包) ============
    print(f"\n数据预处理 (量化 → 舍入 → 补码 → 批量打包)")
    (packed_c1, metadata_c1), preprocess_time_c1 = preprocess_matrices(matrices1)
    print(f"客户端 1: {preprocess_time_c1:.2f} ms")
    print(f"  原始梯度数: {sum([m['original_len'] for m in metadata_c1]):,}")
    print(f"  打包后数: {sum([len(p) for p in packed_c1]):,} (压缩比 {sum([m['original_len'] for m in metadata_c1]) / sum([len(p) for p in packed_c1]):.1f}x)")
    
    (packed_c2, metadata_c2), preprocess_time_c2 = preprocess_matrices(matrices2)
    print(f"客户端 2: {preprocess_time_c2:.2f} ms")
    
    total_preprocess = preprocess_time_c1 + preprocess_time_c2
    print(f"总耗时: {total_preprocess:.2f} ms")
    results['timings']['Preprocess_C1'] = preprocess_time_c1
    results['timings']['Preprocess_C2'] = preprocess_time_c2
    results['timings']['Preprocess_Total'] = total_preprocess
    
    # ============ 步骤 7: 加密 (对打包值加密) ============
    print(f"\n加密 (对打包后的大整数加密)")
    encrypted_c1, encrypt_time_c1 = encrypt_matrices(packed_c1, metadata_c1, encryptor, kit)
    print(f"客户端 1: {encrypt_time_c1:.2f} ms ({len(encrypted_c1)} 个密文)")
    
    encrypted_c2, encrypt_time_c2 = encrypt_matrices(packed_c2, metadata_c2, encryptor, kit)
    print(f"客户端 2: {encrypt_time_c2:.2f} ms ({len(encrypted_c2)} 个密文)")
    
    total_encrypt = encrypt_time_c1 + encrypt_time_c2
    print(f"总耗时: {total_encrypt:.2f} ms")
    results['timings']['Encrypt_C1'] = encrypt_time_c1
    results['timings']['Encrypt_C2'] = encrypt_time_c2
    results['timings']['Encrypt_Total'] = total_encrypt
    
    # ============ 步骤 8: 同态聚合 ============
    print(f"\n同态聚合 (E(g1) + E(g2) = E(g1+g2))")
    aggregated_encrypted, aggregate_time = homomorphic_aggregation(
        [encrypted_c1, encrypted_c2],
        evaluator
    )
    print(f"耗时: {aggregate_time:.2f} ms")
    results['timings']['Aggregation'] = aggregate_time
    
    # ============ 步骤 9-10: 解密 + 解包 + 反量化 ============
    print(f"\n解密 + 解包 + 反量化")
    decrypted_aggregated, decrypt_time = decrypt_matrices(aggregated_encrypted, metadata_c1, decryptor)
    print(f"耗时: {decrypt_time:.2f} ms")
    results['timings']['Decryption'] = decrypt_time
    
    # ============ 精度验证 ============
    print(f"\n精度验证")
    plaintext_sum = [m1 + m2 for m1, m2 in zip(matrices1, matrices2)]
    
    # 展平所有矩阵进行对比
    plaintext_flat = np.concatenate([m.reshape(-1) for m in plaintext_sum])
    decrypted_flat = np.concatenate([m.reshape(-1) for m in decrypted_aggregated])
    
    # 确保长度一致
    min_len = min(len(plaintext_flat), len(decrypted_flat))
    plaintext_flat = plaintext_flat[:min_len]
    decrypted_flat = decrypted_flat[:min_len]
    
    relative_error = np.linalg.norm(decrypted_flat - plaintext_flat) / (np.linalg.norm(plaintext_flat) + 1e-10) * 100
    print(f"相对误差: {relative_error:.6f}%")
    results['accuracy'] = {
        'relative_error_percent': relative_error
    }
    
    # ============ 性能总结 ============
    print(f"\n" + "="*80)
    print("性能总结 - BatchCrypt 完整流程")
    print("="*80)
    
    total_time = keygen_time + total_preprocess + total_encrypt + aggregate_time + decrypt_time
    
    print(f"\n各阶段耗时 (毫秒):")
    print(f" 密钥生成:        {keygen_time:10.2f} ms")
    print(f" 数据预处理:      {total_preprocess:10.2f} ms (量化+舍入+补码+打包)")
    print(f" 加密 (打包值):   {total_encrypt:10.2f} ms ({total_encrypt/total_time*100:.1f}%)")
    print(f" 同态聚合:        {aggregate_time:10.2f} ms")
    print(f" 解密+恢复:       {decrypt_time:10.2f} ms ({decrypt_time/total_time*100:.1f}%)")
    print(f"   " + "-"*50)
    print(f" 总耗时 (不含KeyGen): {total_time - keygen_time:10.2f} ms")
    
    # 计算比例
    preprocess_ratio = total_preprocess / (total_time - keygen_time) * 100
    encrypt_ratio = total_encrypt / (total_time - keygen_time) * 100
    aggregate_ratio = aggregate_time / (total_time - keygen_time) * 100
    decrypt_ratio = decrypt_time / (total_time - keygen_time) * 100
    
    print(f"\n占比 (不含KeyGen):")
    print(f" 数据预处理:      {preprocess_ratio:6.1f}%")
    print(f" 加密:            {encrypt_ratio:6.1f}%")
    print(f" 同态聚合:        {aggregate_ratio:6.1f}%")
    print(f" 解密+恢复:       {decrypt_ratio:6.1f}%")
    
    print(f"\n精度验证:")
    print(f" 相对误差: {relative_error:.6f}% {'✓ 精度良好' if relative_error < 1.0 else '⚠ 精度损失较大'}")
    
    # BatchCrypt 优化效果
    original_gradient_count = sum([m['original_len'] for m in metadata_c1])
    packed_count = sum([len(p) for p in packed_c1])
    speedup_ratio = original_gradient_count / packed_count
    
    print(f"\nBatchCrypt 优化效果:")
    print(f" 原始梯度数量:    {original_gradient_count:,}")
    print(f" 打包后密文数:    {packed_count:,}")
    print(f" 理论加速比:      {speedup_ratio:.1f}x (批量打包优化)")
    print(f" 量化位宽:        8 位 (32位浮点 → 8位整数)")
    print(f" 批量大小:        16 个梯度/密文")
    
    # 输出 FLOPs 统计
    flops_counter.report()
    
    # 保存结果（转换numpy类型为Python原生类型）
    def convert_to_serializable(obj):
        """递归转换numpy类型为Python原生类型"""
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = f'batchcrypt_zpaillier_result_{timestamp}.json'
    with open(result_file, 'w') as f:
        json.dump(convert_to_serializable(results), f, indent=2)
    print(f"\n 结果已保存: {result_file}")
    
    return results


# ============ 参数对比测试 ============

def parameter_comparison(file_path1='./cnn/cnn.npz', file_path2='./cnn/cnn1.npz'):
    """
    测试不同密钥长度的性能差异（完整 BatchCrypt 流程）
    """
    print("\n" + "="*80)
    print("🔬 参数对比测试 - 密钥长度影响 (BatchCrypt完整流程)")
    print("="*80)
    
    key_lengths = [1024, 2048, 4096]
    results_list = []
    
    for key_length in key_lengths:
        print(f"\n【测试密钥长度: {key_length} bits】")
        results = run_benchmark(file_path1, file_path2, key_length)
        if results:
            results_list.append(results)
    
    # 对比输出
    print(f"\n" + "="*80)
    print(" 参数对比结果")
    print("="*80)
    print(f"\n{'Key Length':<12} {'KeyGen':<12} {'Preprocess':<12} {'Encrypt':<12} {'Aggregate':<12} {'Decrypt':<12}")
    print("-" * 72)
    
    for results in results_list:
        key_length = results['key_length']
        keygen = results['timings'].get('KeyGen', 0)
        preprocess = results['timings'].get('Preprocess_Total', 0)
        encrypt = results['timings'].get('Encrypt_Total', 0)
        aggregate = results['timings'].get('Aggregation', 0)
        decrypt = results['timings'].get('Decryption', 0)
        
        print(f"{key_length:<12} {keygen:<12.2f} {preprocess:<12.2f} {encrypt:<12.2f} {aggregate:<12.2f} {decrypt:<12.2f}")
    
    return results_list


# ============ FLOPs 理论统计 ============
class FlopsCounter:
    def __init__(self):
        self.preprocess = 0  # 量化、舍入、转换
        self.encrypt = 0
        self.decrypt = 0
        self.aggregate = 0
        self.postprocess = 0  # 解包、反量化
    
    def total(self):
        return self.preprocess + self.encrypt + self.decrypt + self.aggregate + self.postprocess
    
    def report(self):
        print(f"\n[理论 FLOPs 统计]")
        print(f"  预处理 FLOPs:   {self.preprocess:,}")
        print(f"  加密 FLOPs:     {self.encrypt:,}")
        print(f"  聚合 FLOPs:     {self.aggregate:,}")
        print(f"  解密 FLOPs:     {self.decrypt:,}")
        print(f"  后处理 FLOPs:   {self.postprocess:,}")
        print(f"  总 FLOPs:       {self.total():,}")

# 初始化 FLOPs 计数器
flops_counter = FlopsCounter()


# ============ 主函数 ============

def main():
    """主程序入口"""
    print("\n╔════════════════════════════════════════════════════════════════════════════════╗")
    print("║          BatchCrypt 性能测试 (完整量化+打包流程 + ZPaillier)                  ║")
    print("║                                                                                ║")
    print("║ 核心优化:                                                                      ║")
    print("║   ✓ 梯度量化 (32位 → 8位)                                                     ║")
    print("║   ✓ 随机舍入 (保持无偏性)                                                     ║")
    print("║   ✓ 批量打包 (16个梯度 → 1个密文, 16x加速)                                   ║")
    print("║   ✓ 同态加密 (ZPaillier)                                                      ║")
    print("║                                                                                ║")
    print("║ 选项:                                                                          ║")
    print("║   1. 单一配置测试 (推荐首先尝试)                                               ║")
    print("║   2. 参数对比测试 (多种密钥长度)                                               ║")
    print("║   3. 退出                                                                      ║")
    print("╚════════════════════════════════════════════════════════════════════════════════╝\n")
    
    choice = input("请选择 (1-3): ").strip()
    
    if choice == '1':
        # 单一配置测试
        results = run_benchmark(
            file_path1='./cnn/cnn.npz',
            file_path2='./cnn/cnn1.npz',
            key_length=2048
        )
        if results:
            print("\n 测试完成！")
    
    elif choice == '2':
        # 参数对比测试
        results_list = parameter_comparison(
            file_path1='./cnn/cnn.npz',
            file_path2='./cnn/cnn1.npz'
        )
        print("\n 对比测试完成！")
    
    elif choice == '3':
        print("退出")
    
    else:
        print("无效选择")


if __name__ == "__main__":
    main()
