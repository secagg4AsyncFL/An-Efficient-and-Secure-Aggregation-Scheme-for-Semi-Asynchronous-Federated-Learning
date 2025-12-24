from heu import numpy as hnp
from heu import phe
import numpy as np

from functools import wraps
import time


def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"{func.__name__} 执行时间: {execution_time:.4f} 秒")
        return result
    return wrapper

OU_kit = hnp.setup(phe.SchemaType.OU, 2048)
OU_encryptor = OU_kit.encryptor()
OU_decryptor = OU_kit.decryptor()
OU_evaluator = OU_kit.evaluator()

ZP_kit = hnp.setup(phe.SchemaType.ZPaillier, 2048)
ZP_encryptor = ZP_kit.encryptor()
ZP_decryptor = ZP_kit.decryptor()
ZP_evaluator = ZP_kit.evaluator()


def load_matrices_from_npz(file_path):
    with np.load(file_path) as data:
        matrices = [data[key] for key in data.files]
    return matrices




# ============ FLOPs 理论统计 ============
class FlopsCounter:
    def __init__(self, name=""):
        self.name = name
        self.encrypt = 0
        self.decrypt = 0
        self.aggregate = 0
    def total(self):
        return self.encrypt + self.decrypt + self.aggregate
    def report(self):
        print(f"\n[理论 FLOPs 统计 - {self.name}]")
        print(f"  加密 FLOPs:     {self.encrypt:,}")
        print(f"  解密 FLOPs:     {self.decrypt:,}")
        print(f"  聚合 FLOPs:     {self.aggregate:,}")
        print(f"  总 FLOPs:       {self.total():,}")

# 分别为 OU 和 ZPaillier 创建计数器
flops_counter_OU = FlopsCounter("OU")
flops_counter_ZP = FlopsCounter("ZPaillier")


@timing_decorator
def encrypt_matrices(matrices, encryptor, kit, flops_counter=None):
    encrypted_matrices = []
    for matrix in matrices:
        tempArr = kit.array(matrix.reshape(-1), phe.FloatEncoderParams())
        encrypted_matrix = encryptor.encrypt(tempArr)
        encrypted_matrices.append(encrypted_matrix)
        # FLOPs统计：每个元素加密约2次乘法
        if flops_counter:
            flops_counter.encrypt += 2 * tempArr.size
    return encrypted_matrices


@timing_decorator
def decrypt_matrix(encrypted_matrix, decryptor, flops_counter=None):
    decrypted = decryptor.decrypt(encrypted_matrix)
    # FLOPs统计：每个元素解密约2次乘法
    if flops_counter:
        if hasattr(decrypted, 'size'):
            flops_counter.decrypt += 2 * decrypted.size
        elif isinstance(decrypted, np.ndarray):
            flops_counter.decrypt += 2 * decrypted.size
    return decrypted


@timing_decorator
def homomorphic_addition(encrypted_matrix1, encrypted_matrix2, evaluator, flops_counter=None):
    result = evaluator.add(encrypted_matrix1, encrypted_matrix2)
    # FLOPs统计：每个元素加法1次
    if flops_counter:
        if hasattr(encrypted_matrix1, 'size'):
            flops_counter.aggregate += encrypted_matrix1.size
        elif isinstance(encrypted_matrix1, np.ndarray):
            flops_counter.aggregate += encrypted_matrix1.size
    return result



def main():
    
    file_path1 = './cnn/cnn.npz'  
    file_path2 = './cnn/cnn1.npz' 

    matrices1 = load_matrices_from_npz(file_path1)
    matrices2 = load_matrices_from_npz(file_path2)
    

    start_time = time.time()    
    OU_encrypted_matrices1 = encrypt_matrices(matrices1, OU_encryptor, OU_kit, flops_counter_OU)
    OU_encrypted_matrices2 = encrypt_matrices(matrices2, OU_encryptor, OU_kit, flops_counter_OU)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"OU加密时间: {execution_time:.4f} 秒")

    start_time = time.time()    
    ZP_encrypted_matrices1 = encrypt_matrices(matrices1, ZP_encryptor, ZP_kit, flops_counter_ZP)
    ZP_encrypted_matrices2 = encrypt_matrices(matrices2, ZP_encryptor, ZP_kit, flops_counter_ZP)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"ZPaillier加密时间: {execution_time:.4f} 秒")


    start_time = time.time()
    OU_result_encrypted_matrices = []
    for enc_mat1, enc_mat2 in zip(OU_encrypted_matrices1, OU_encrypted_matrices2):
        OU_result_encrypted = homomorphic_addition(enc_mat1, enc_mat2, OU_evaluator, flops_counter_OU)
        OU_result_encrypted_matrices.append(OU_result_encrypted)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"OU聚合时间: {execution_time:.4f} 秒")

    start_time = time.time()
    ZP_result_encrypted_matrices = []
    for enc_mat1, enc_mat2 in zip(ZP_encrypted_matrices1, ZP_encrypted_matrices2):
        ZP_result_encrypted = homomorphic_addition(enc_mat1, enc_mat2, ZP_evaluator, flops_counter_ZP)
        ZP_result_encrypted_matrices.append(ZP_result_encrypted)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"ZPaillier聚合时间: {execution_time:.4f} 秒")


    start_time = time.time()
    OU_decrypted_matrices = [decrypt_matrix(enc_mat, OU_decryptor, flops_counter_OU) for enc_mat in OU_result_encrypted_matrices]
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"OU解密时间: {execution_time:.4f} 秒")

    start_time = time.time()
    ZP_decrypted_matrices = [decrypt_matrix(enc_mat, ZP_decryptor, flops_counter_ZP) for enc_mat in ZP_result_encrypted_matrices]
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"ZPaillier解密时间: {execution_time:.4f} 秒")


    # 分别输出 OU 和 ZPaillier 的 FLOPs 统计
    flops_counter_OU.report()
    flops_counter_ZP.report()


if __name__ == "__main__":
    main()
















