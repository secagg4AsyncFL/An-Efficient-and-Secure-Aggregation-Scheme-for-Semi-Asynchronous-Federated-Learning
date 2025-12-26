import numpy as np
import tenseal as ts

from functools import wraps
import time


# 设置TenSEAL上下文
context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60,40,40,60])
context.generate_galois_keys()
context.global_scale = 2**40


# ============ FLOPs 理论统计 ============
class FlopsCounter:
    def __init__(self):
        self.encrypt = 0
        self.decrypt = 0
        self.aggregate = 0
    def total(self):
        return self.encrypt + self.decrypt + self.aggregate
    def report(self):
        print(f"\n[Theoretical FLOPs Statistics]")
        print(f"  Encryption FLOPs:     {self.encrypt:,}")
        print(f"  Decryption FLOPs:     {self.decrypt:,}")
        print(f"  Aggregation FLOPs:     {self.aggregate:,}")
        print(f"  Total FLOPs:       {self.total():,}")

flops_counter = FlopsCounter()


def load_matrices_from_npz(file_path):
    with np.load(file_path) as data:
        matrices = [data[key] for key in data.files]
    return matrices


def encrypt_matrices(matrices, context):
    encrypted_matrices = []
    for matrix in matrices:
        encrypted_matrix = ts.ckks_vector(context, matrix.reshape(-1))
        encrypted_matrices.append(encrypted_matrix)
        flops_counter.encrypt += 2 * matrix.size
    return encrypted_matrices


def decrypt_matrix(encrypted_matrix):
    decrypted = encrypted_matrix.decrypt()
    if hasattr(decrypted, '__len__'):
        flops_counter.decrypt += 2 * len(decrypted)
    return decrypted


def homomorphic_addition(encrypted_matrix1, encrypted_matrix2):
    result = encrypted_matrix1 + encrypted_matrix2
    if hasattr(encrypted_matrix1, 'size'):
        flops_counter.aggregate += encrypted_matrix1.size
    return result



def main():

    file_path1 = './cnn/cnn.npz'  
    file_path2 = './cnn/cnn1.npz'  

    matrices1 = load_matrices_from_npz(file_path1)
    matrices2 = load_matrices_from_npz(file_path2)

    start_time = time.time()
    encrypted_matrices1 = encrypt_matrices(matrices1, context)
    encrypted_matrices2 = encrypt_matrices(matrices2, context)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Encryption Time: {execution_time:.4f} seconds")


    start_time = time.time()
    result_encrypted_matrices = []
    for enc_mat1, enc_mat2 in zip(encrypted_matrices1, encrypted_matrices2):
        result_encrypted = homomorphic_addition(enc_mat1, enc_mat2)
        result_encrypted_matrices.append(result_encrypted)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Aggregation Time: {execution_time:.4f} seconds")

    start_time = time.time()
    decrypted_matrices = [decrypt_matrix(enc_mat) for enc_mat in result_encrypted_matrices]
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Decryption Time: {execution_time:.4f} seconds")

    # print("\n[Theoretical FLOPs Statistics]")
    # flops_counter.report()

if __name__ == "__main__":
    main()