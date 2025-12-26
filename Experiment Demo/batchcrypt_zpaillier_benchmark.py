
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
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = (end_time - start_time) * 1000  
        return result, execution_time
    return wrapper


# ============ 步骤 1: 密钥生成 ============

@timing_decorator
def key_generation(key_length=2048):
    kit = hnp.setup(phe.SchemaType.ZPaillier, key_length)
    encryptor = kit.encryptor()
    decryptor = kit.decryptor()
    evaluator = kit.evaluator()
    return (encryptor, decryptor, evaluator, kit)






def quantize_matrix(matrix, bit_width=8, r_max=0.5):

    og_sign = np.sign(matrix)
    uns_matrix = matrix * og_sign  
    uns_result = uns_matrix * (pow(2, bit_width - 1) - 1.0) / r_max
    result = og_sign * uns_result
    return result


# ============ 步骤 3: 随机舍入 ============

def stochastic_round(ori):

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

    _shape = ori.shape
    ori = ori.reshape(-1)
    result = stochastic_round(ori)
    result = result.reshape(_shape)
    return result



def true_to_two_comp(input_array, bit_width):

    result = np.zeros(len(input_array), dtype=np.int32)
    for i in range(len(input_array)):
        if input_array[i] >= 0:
            result[i] = input_array[i]
        else:
            result[i] = 2 ** (bit_width + 1) + input_array[i]
    return result



def batch_pack(array, batch_size=16, bit_width=8, pad_zero=3):

    A_len = len(array)
    if (A_len % batch_size) != 0:
        array = np.pad(array, (0, batch_size - (A_len % batch_size)), 'constant', constant_values=0)
    
    idx_range = int(len(array) / batch_size)
    idx_base = list(range(idx_range))
    batched_nums = np.zeros(idx_range, dtype=object)  
    
    for i in range(batch_size):
        idx_filter = [i + x * batch_size for x in idx_base]
        filtered_num = array[idx_filter]
        batched_nums = (batched_nums * pow(2, (bit_width + pad_zero))) + filtered_num
    
    return batched_nums, A_len



@timing_decorator
def preprocess_matrices(matrices, bit_width=8, r_max=0.5, batch_size=16, pad_zero=3):

    processed = []
    metadata = []
    
    for matrix in matrices:
        og_shape = matrix.shape
        flattened = matrix.reshape(-1)
        
        quantized = quantize_matrix(flattened, bit_width, r_max)
        
        rounded = stochastic_round(quantized)
        
        two_comp = true_to_two_comp(rounded, bit_width)
        
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
        
        flops_counter.preprocess += 3 * original_len
    
    return processed, metadata



@timing_decorator
def encrypt_matrices(packed_matrices_list, metadata_list, encryptor, kit):

    encrypted_matrices = []
    
    for packed_array, metadata in zip(packed_matrices_list, metadata_list):
        packed_ints = []
        for packed_value in packed_array:
            if isinstance(packed_value, np.ndarray):
                packed_ints.append(int(packed_value.item()))
            else:
                packed_ints.append(int(packed_value))
        
        encoder = phe.BigintEncoder(phe.SchemaType.ZPaillier)
        heu_plaintext = kit.array(packed_ints, encoder)
        encrypted_array = encryptor.encrypt(heu_plaintext)
        
        encrypted_matrices.append(encrypted_array)
        
        flops_counter.encrypt += 2 * len(packed_ints)
    
    return encrypted_matrices




@timing_decorator
def homomorphic_aggregation(encrypted_matrices_list, evaluator):

    aggregated_matrices = []
    for enc_matrix in encrypted_matrices_list[0]:
        aggregated_matrices.append(enc_matrix)
    
    for client_encrypted_matrices in encrypted_matrices_list[1:]:
        for matrix_idx in range(len(aggregated_matrices)):
            aggregated_matrices[matrix_idx] = evaluator.add(
                aggregated_matrices[matrix_idx],
                client_encrypted_matrices[matrix_idx]
            )

            flops_counter.aggregate += aggregated_matrices[matrix_idx].size
    
    return aggregated_matrices



def two_comp_to_true(two_comp, bit_width=8, pad_zero=3):

    if two_comp < 0:
        raise Exception("Error: not expecting negative value")
    
    sign = two_comp >> (bit_width - 1)
    literal = two_comp & (2 ** (bit_width - 1) - 1)
    
    if sign == 0:  
        return literal
    elif sign == 4: 
        return literal
    elif sign == 1: 
        return pow(2, bit_width - 1) - 1
    elif sign == 3:  
        return - 1 * (2 ** (bit_width - 1) - literal)
    elif sign == 7: 
        return - 1 * (2 ** (bit_width - 1) - literal)
    elif sign == 6:  
        print(f'  Warning: Negative overflow {two_comp}')
        return - (pow(2, bit_width - 1) - 1)
    else:  
        print(f'  Warning: Unrecognized overflow {two_comp}')
        return - (pow(2, bit_width - 1) - 1)


def unpack_batch(packed_array, original_len, batch_size=16, bit_width=8, pad_zero=3):

    num_ele_w_pad = batch_size * len(packed_array)
    un_batched_nums = np.zeros(num_ele_w_pad, dtype=np.int32)
    
    for i in range(batch_size):
        filter_mask = (pow(2, bit_width + pad_zero) - 1) << ((bit_width + pad_zero) * i)
        
        for j in range(len(packed_array)):
            packed_int = int(packed_array[j])  
            two_comp = (filter_mask & packed_int) >> ((bit_width + pad_zero) * i)
            un_batched_nums[batch_size * j + batch_size - 1 - i] = two_comp_to_true(two_comp, bit_width, pad_zero)
    
    un_batched_nums = un_batched_nums[:original_len]
    return un_batched_nums



def unquantize_matrix(matrix, bit_width=8, r_max=0.5):

    matrix = matrix.astype(np.int32)
    og_sign = np.sign(matrix)
    uns_matrix = matrix * og_sign
    uns_result = uns_matrix * r_max / (pow(2, bit_width - 1) - 1.0)
    result = og_sign * uns_result
    return result.astype(np.float32)



@timing_decorator
def decrypt_matrices(encrypted_matrices, metadata_list, decryptor):

    decrypted_matrices = []
    
    for encrypted_array, metadata in zip(encrypted_matrices, metadata_list):
        try:
            dec_array = decryptor.decrypt(encrypted_array)
            arr_size = encrypted_array.size
            decrypted_packed = [int(dec_array[i]) for i in range(arr_size)]
            flops_counter.decrypt += 2 * len(decrypted_packed)
        except Exception as e:
            print(f"  Decrypt Error: {e}")
            arr_size = encrypted_array.size
            decrypted_packed = [0] * arr_size
            flops_counter.decrypt += 2 * arr_size
        
        unpacked = unpack_batch(
            decrypted_packed,
            metadata['original_len'],
            metadata['batch_size'],
            metadata['bit_width'],
            metadata['pad_zero']
        )
        
        unquantized = unquantize_matrix(
            unpacked,
            metadata['bit_width'],
            metadata['r_max']
        )
        
        restored = unquantized.reshape(metadata['shape'])
        decrypted_matrices.append(restored)
        
        flops_counter.postprocess += 2 * metadata['original_len']
    
    return decrypted_matrices




def load_matrices_from_npz(file_path):
    with np.load(file_path) as data:
        matrices = [data[key] for key in sorted(data.files)]
    return matrices



def run_benchmark(file_path1='./cnn/cnn.npz', file_path2='./cnn/cnn1.npz', key_length=2048):

    
    global flops_counter
    flops_counter = FlopsCounter()
    
    print("\n" + "="*80)
    print("BatchCrypt Full Pipeline Performance Test (Quantization + Packing + ZPaillier)")
    print("="*80)
    
    # Check if files exist
    if not Path(file_path1).exists() or not Path(file_path2).exists():
        print(f"Data files do not exist: {file_path1} or {file_path2}")
        return None
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'key_length': key_length,
        'timings': {}
    }
    
    print(f"\nKey Generation (key_length={key_length})")
    (encryptor, decryptor, evaluator, kit), keygen_time = key_generation(key_length)
    print(f"Time: {keygen_time:.2f} ms")
    results['timings']['KeyGen'] = keygen_time
    
    # ============ Load Data ============
    print(f"\nLoading data")
    matrices1 = load_matrices_from_npz(file_path1)
    matrices2 = load_matrices_from_npz(file_path2)
    print(f"Client 1: {len(matrices1)} matrices")
    print(f"Client 2: {len(matrices2)} matrices")
    
    print(f"\nData Preprocessing (Quantization → Rounding → Two's Complement → Batch Packing)")
    (packed_c1, metadata_c1), preprocess_time_c1 = preprocess_matrices(matrices1)
    print(f"Client 1: {preprocess_time_c1:.2f} ms")
    print(f"  Original Gradient Count: {sum([m['original_len'] for m in metadata_c1]):,}")
    print(f"  Packed Count: {sum([len(p) for p in packed_c1]):,} (Compression Ratio {sum([m['original_len'] for m in metadata_c1]) / sum([len(p) for p in packed_c1]):.1f}x)")
    
    (packed_c2, metadata_c2), preprocess_time_c2 = preprocess_matrices(matrices2)
    print(f"Client 2: {preprocess_time_c2:.2f} ms")
    
    total_preprocess = preprocess_time_c1 + preprocess_time_c2
    print(f"Total Time: {total_preprocess:.2f} ms")
    results['timings']['Preprocess_C1'] = preprocess_time_c1
    results['timings']['Preprocess_C2'] = preprocess_time_c2
    results['timings']['Preprocess_Total'] = total_preprocess
    
    print(f"\nEncryption (Encrypting Packed Large Integers with ZPaillier)")
    encrypted_c1, encrypt_time_c1 = encrypt_matrices(packed_c1, metadata_c1, encryptor, kit)
    print(f"Client 1: {encrypt_time_c1:.2f} ms ({len(encrypted_c1)} ciphertexts)")
    
    encrypted_c2, encrypt_time_c2 = encrypt_matrices(packed_c2, metadata_c2, encryptor, kit)
    print(f"Client 2: {encrypt_time_c2:.2f} ms ({len(encrypted_c2)} ciphertexts)")
    
    total_encrypt = encrypt_time_c1 + encrypt_time_c2
    print(f"Total Time: {total_encrypt:.2f} ms")
    results['timings']['Encrypt_C1'] = encrypt_time_c1
    results['timings']['Encrypt_C2'] = encrypt_time_c2
    results['timings']['Encrypt_Total'] = total_encrypt
    
    print(f"\nHomomorphic Aggregation (E(g1) + E(g2) = E(g1+g2))")
    aggregated_encrypted, aggregate_time = homomorphic_aggregation(
        [encrypted_c1, encrypted_c2],
        evaluator
    )
    print(f"Time: {aggregate_time:.2f} ms")
    results['timings']['Aggregation'] = aggregate_time
    
    print(f"\nDecryption + Unpacking + Dequantization")
    decrypted_aggregated, decrypt_time = decrypt_matrices(aggregated_encrypted, metadata_c1, decryptor)
    print(f"Time: {decrypt_time:.2f} ms")
    results['timings']['Decryption'] = decrypt_time
    
    print(f"\nAccuracy Verification")
    plaintext_sum = [m1 + m2 for m1, m2 in zip(matrices1, matrices2)]
    
    plaintext_flat = np.concatenate([m.reshape(-1) for m in plaintext_sum])
    decrypted_flat = np.concatenate([m.reshape(-1) for m in decrypted_aggregated])
    
    min_len = min(len(plaintext_flat), len(decrypted_flat))
    plaintext_flat = plaintext_flat[:min_len]
    decrypted_flat = decrypted_flat[:min_len]
    
    relative_error = np.linalg.norm(decrypted_flat - plaintext_flat) / (np.linalg.norm(plaintext_flat) + 1e-10) * 100
    print(f"Relative Error: {relative_error:.6f}%")
    results['accuracy'] = {
        'relative_error_percent': relative_error
    }
    
    print(f"\n" + "="*80)
    print("Performance Summary - BatchCrypt Full Process")
    print("="*80)
    
    total_time = keygen_time + total_preprocess + total_encrypt + aggregate_time + decrypt_time
    
    print(f"\nTimings (milliseconds):")
    print(f" Key Generation:        {keygen_time:10.2f} ms")
    print(f" Data Preprocessing:      {total_preprocess:10.2f} ms (Quantization + Rounding + Two's Complement + Packing)")
    print(f" Encryption (Packed Values):   {total_encrypt:10.2f} ms ({total_encrypt/total_time*100:.1f}%)")
    print(f" Homomorphic Aggregation:        {aggregate_time:10.2f} ms")
    print(f" Decryption + Recovery:       {decrypt_time:10.2f} ms ({decrypt_time/total_time*100:.1f}%)")
    print(f"   " + "-"*50)
    print(f" Total Time (excluding KeyGen): {total_time - keygen_time:10.2f} ms")
    
    preprocess_ratio = total_preprocess / (total_time - keygen_time) * 100
    encrypt_ratio = total_encrypt / (total_time - keygen_time) * 100
    aggregate_ratio = aggregate_time / (total_time - keygen_time) * 100
    decrypt_ratio = decrypt_time / (total_time - keygen_time) * 100
    
    print(f"\nProportions (excluding KeyGen):")
    print(f" Data Preprocessing:      {preprocess_ratio:6.1f}%")
    print(f" Encryption:            {encrypt_ratio:6.1f}%")
    print(f" Homomorphic Aggregation:        {aggregate_ratio:6.1f}%")
    print(f" Decryption + Recovery:       {decrypt_ratio:6.1f}%")
    
    print(f"\nAccuracy Verification:")
    print(f" Relative Error: {relative_error:.6f}% {'Good Accuracy' if relative_error < 1.0 else '⚠ Significant Accuracy Loss'}")
    
    original_gradient_count = sum([m['original_len'] for m in metadata_c1])
    packed_count = sum([len(p) for p in packed_c1])
    speedup_ratio = original_gradient_count / packed_count
    
    print(f"\nBatchCrypt Optimization Effect:")
    print(f" Original Gradient Count:    {original_gradient_count:,}")
    print(f" Packed Ciphertext Count:    {packed_count:,}")
    print(f" Theoretical Speedup:      {speedup_ratio:.1f}x (Batch Packing Optimization)")
    print(f" Quantization Bit-Width:        8 bits (32-bit float → 8-bit integer)")
    print(f" Batch Size:        16 gradients/ciphertext")
    
    # flops_counter.report()
    
    def convert_to_serializable(obj):
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
    print(f"\n Results saved: {result_file}")
    
    return results



def parameter_comparison(file_path1='./cnn/cnn.npz', file_path2='./cnn/cnn1.npz'):

    print("\n" + "="*80)
    print("Parameter Comparison Test - Impact of Key Length (Full BatchCrypt Process)")
    print("="*80)
    
    key_lengths = [1024, 2048, 4096]
    results_list = []
    
    for key_length in key_lengths:
        print(f"\n【Testing Key Length: {key_length} bits】")
        results = run_benchmark(file_path1, file_path2, key_length)
        if results:
            results_list.append(results)
    

    print(f"\n" + "="*80)
    print(" Parameter Comparison Results")
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


class FlopsCounter:
    def __init__(self):
        self.preprocess = 0  
        self.encrypt = 0
        self.decrypt = 0
        self.aggregate = 0
        self.postprocess = 0  
    
    def total(self):
        return self.preprocess + self.encrypt + self.decrypt + self.aggregate + self.postprocess
    
    def report(self):
        print(f"\n[Theoretical FLOPs Statistics]")
        print(f"  Preprocessing FLOPs:   {self.preprocess:,}")
        print(f"  Encryption FLOPs:     {self.encrypt:,}")
        print(f"  Aggregation FLOPs:     {self.aggregate:,}")
        print(f"  Decryption FLOPs:     {self.decrypt:,}")
        print(f"  Postprocessing FLOPs:   {self.postprocess:,}")
        print(f"  Total FLOPs:       {self.total():,}")

flops_counter = FlopsCounter()



def main():
    
    # choice = input("Please choose (1-3): ").strip()
    choice = '1'
    
    if choice == '1':
        results = run_benchmark(
            file_path1='./cnn/cnn.npz',
            file_path2='./cnn/cnn1.npz',
            key_length=2048
        )
        if results:
            print("\n Test completed!")
    
    elif choice == '2':
        # Parameter comparison test
        results_list = parameter_comparison(
            file_path1='./cnn/cnn.npz',
            file_path2='./cnn/cnn1.npz'
        )
        print("\n Parameter comparison test completed!")
    
    elif choice == '3':
        print("Exit")
    
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()
