import numpy as np
import time
from typing import List, Tuple
from sympy import primefactors, totient
from numpy.polynomial import polynomial as P
import os


class IntCRT:
    
    def __init__(self, moduli: List[int]):

        self.moduli = moduli
        self.M = np.prod(moduli)
        self.m_prime = [self.M // m for m in moduli]
        self.t = [self._modinv(mp, m) for mp, m in zip(self.m_prime, moduli)]
    
    def _modinv(self, a: int, m: int) -> int:
        return pow(int(a), -1, int(m))

    def encode(self, values: List[int]) -> int:
       
        assert len(values) == len(self.moduli), "Number of values must match moduli"
        
        result = 0
        for v, t, mp in zip(values, self.t, self.m_prime):
            result += (v % self.moduli[values.index(v)]) * t * mp
        
        return result % self.M
    
    def decode(self, packed: int) -> List[int]:
       
        return [packed % m for m in self.moduli]


class IntCat:
    
    def __init__(self, base: int, num_values: int):
        self.base = base
        self.num_values = num_values
    
    def encode(self, values: List[int]) -> int:

        assert len(values) == self.num_values
        
        result = 0
        for i, v in enumerate(values):
            result += v * (self.base ** i)
        
        return result
    
    def decode(self, packed: int) -> List[int]:

        values = []
        for _ in range(self.num_values):
            values.append(packed % self.base)
            packed //= self.base
        
        return values


class PolyCoef:
    
    def __init__(self, degree: int, modulus: int):

        self.degree = degree
        self.modulus = modulus
    
    def encode(self, values: List[int]) -> np.ndarray:

        assert len(values) <= self.degree
        
        poly = np.zeros(self.degree, dtype=np.int64)
        poly[:len(values)] = values
        
        return poly % self.modulus
    
    def decode(self, poly: np.ndarray) -> List[int]:
        return poly.tolist()


class PolySubR:
    
    def __init__(self, degree: int, modulus: int):
        assert degree & (degree - 1) == 0, "Degree must be power of 2"
        
        self.degree = degree
        self.modulus = modulus
        self.num_slots = degree  # Simplified: assuming m | p-1
    
    def _ntt_forward(self, poly: np.ndarray) -> np.ndarray:
        # Simplified NTT for demonstration
        return np.fft.fft(poly).real.astype(np.int64) % self.modulus
    
    def _ntt_inverse(self, slots: np.ndarray) -> np.ndarray:
        # Simplified inverse NTT
        return np.fft.ifft(slots).real.astype(np.int64) % self.modulus
    
    def encode(self, values: List[int]) -> np.ndarray:

        assert len(values) <= self.num_slots
        
        slots = np.zeros(self.num_slots, dtype=np.int64)
        slots[:len(values)] = values
        
        # Apply inverse NTT to get polynomial representation
        poly = self._ntt_inverse(slots)
        
        return poly % self.modulus
    
    def decode(self, poly: np.ndarray) -> List[int]:

        # Apply NTT to get slot representation
        slots = self._ntt_forward(poly)
        
        return slots.tolist()


class RLWEEncryption:
    
    def __init__(self, poly_degree: int, plain_modulus: int, cipher_modulus: int, 
                 error_std: float = 3.2):

        self.N = poly_degree
        self.p = plain_modulus
        self.q = cipher_modulus
        self.error_std = error_std
        
        # Generate secret key
        self.s = self._sample_ternary(poly_degree)
        
        # Generate public key
        self.pk = self._keygen()
    
    def _sample_ternary(self, size: int) -> np.ndarray:
        return np.random.choice([-1, 0, 1], size=size)
    
    def _sample_error(self, size: int) -> np.ndarray:
        return np.round(np.random.normal(0, self.error_std, size=size)).astype(np.int64)
    
    def _poly_mult(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        # Multiply polynomials
        result = np.convolve(a, b)
        
        # Reduce modulo (x^N + 1): x^N = -1
        for i in range(len(result) - 1, self.N - 1, -1):
            result[i - self.N] -= result[i]
        
        result = result[:self.N] % self.q
        
        return result
    
    def _poly_add(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return (a + b) % self.q
    
    def _keygen(self) -> Tuple[np.ndarray, np.ndarray]:

        a = np.random.randint(0, self.q, size=self.N, dtype=np.int64)
        e = self._sample_error(self.N)
        
        b_0 = (self._poly_mult(a, self.s) + self.p * e) % self.q
        b_1 = (-a) % self.q
        
        return (b_0, b_1)
    
    def encrypt(self, plaintext_poly: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        b_0, b_1 = self.pk
        
        # Sample random polynomials
        u = self._sample_error(self.N)
        e_1 = self._sample_error(self.N)
        e_2 = self._sample_error(self.N)
        
        # Compute ciphertext
        c_0 = (self._poly_mult(b_0, u) + self.p * e_1 + plaintext_poly) % self.q
        c_1 = (self._poly_mult(b_1, u) + self.p * e_2) % self.q
        
        return (c_0, c_1)
    
    def decrypt(self, ciphertext: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        
        c_0, c_1 = ciphertext
        
        # Compute m* = (c_0 + c_1 * s mod q) mod p
        m_star = (c_0 + self._poly_mult(c_1, self.s)) % self.q
        m_star = m_star % self.p
        
        return m_star
    
    def add_ciphertexts(self, ct1: Tuple[np.ndarray, np.ndarray], 
                       ct2: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        
        c_0_1, c_1_1 = ct1
        c_0_2, c_1_2 = ct2
        
        return (self._poly_add(c_0_1, c_0_2), self._poly_add(c_1_1, c_1_2))


def load_gradients_from_npz(num_users, gradient_size):

    user_gradients = []
    
    # 查找cnn目录下的梯度文件
    cnn_dir = "./cnn"
    resnet_dir = "./resnet"
    
    # 首先尝试从cnn目录加载
    if os.path.exists(cnn_dir) and os.listdir(cnn_dir):
        grad_files = [f for f in os.listdir(cnn_dir) if f.startswith('cnn') and f.endswith('.npz') and f != 'cnn_aggr.npz']
        grad_files.sort(key=lambda x: int(x[3:].split('.')[0]) if x[3:].split('.')[0].isdigit() else 0)
        
        print(f"Found {len(grad_files)} gradient files in {cnn_dir}")
        
        for i in range(min(num_users, len(grad_files))):
            file_path = os.path.join(cnn_dir, grad_files[i])
            try:
                data = np.load(file_path)
                if 'gradient' in data:
                    grad = data['gradient']
                else:
                    keys = list(data.keys())
                    if keys:
                        grad = data[keys[0]]
                    else:
                        raise ValueError("No arrays found in the .npz file")
                
                if grad.ndim > 1:
                    grad = grad.flatten()
                
                if len(grad) > gradient_size:
                    grad = grad[:gradient_size]
                elif len(grad) < gradient_size:
                    grad = np.pad(grad, (0, gradient_size - len(grad)), mode='constant')
                
                user_gradients.append(grad.astype(np.float64))
                print(f"Loaded gradient from {file_path}, shape: {grad.shape}")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                user_gradients.append(np.random.randn(gradient_size) * 0.01)
    else:
        print("No .npz files found in ./cnn, generating random gradients...")
        user_gradients = [np.random.randn(gradient_size) * 0.01 for _ in range(num_users)]
    
    while len(user_gradients) < num_users:
        print(f"Generating additional random gradient for user {len(user_gradients)+1}")
        user_gradients.append(np.random.randn(gradient_size) * 0.01)
    
    return user_gradients


class TwoPack:
    
    def __init__(self, 
                 int_encoder_type: str = 'crt',
                 poly_encoder_type: str = 'subr',
                 int_batch_size: int = 2,
                 poly_degree: int = 2048,
                 plain_modulus: int = None,
                 cipher_modulus: int = None):

        self.int_batch_size = int_batch_size
        self.poly_degree = poly_degree
        
        # Set default moduli
        if plain_modulus is None:
            plain_modulus = 2**19  # 19-bit as in paper
        if cipher_modulus is None:
            cipher_modulus = 2**60  # Large modulus for security
        
        # Initialize integer encoder
        if int_encoder_type == 'crt':
            moduli = [257, 251, 241, 239][:int_batch_size]  # Small primes
            self.int_encoder = IntCRT(moduli)
        elif int_encoder_type == 'cat':
            base = 2**16  # 16-bit base
            self.int_encoder = IntCat(base, int_batch_size)
        else:
            raise ValueError(f"Unknown int_encoder_type: {int_encoder_type}")
        
        if poly_encoder_type == 'coef':
            self.poly_encoder = PolyCoef(poly_degree, plain_modulus)
        elif poly_encoder_type == 'subr':
            self.poly_encoder = PolySubR(poly_degree, plain_modulus)
        else:
            raise ValueError(f"Unknown poly_encoder_type: {poly_encoder_type}")
        
        self.rlwe = RLWEEncryption(poly_degree, plain_modulus, cipher_modulus)
        
        # self.flops_counter = self.FlopsCounter()
    
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

    def pack_and_encrypt(self, gradients: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        # Quantize gradients to integers (16-bit quantization as in paper)
        scale = 2**15
        int_gradients = np.round(gradients * scale).astype(np.int64)
        
        ciphertexts = []
        
        # Process in batches
        total_values = len(int_gradients)
        values_per_poly = self.poly_degree * self.int_batch_size
        
        for i in range(0, total_values, values_per_poly):
            batch = int_gradients[i:i + values_per_poly]
            
            # Pad if necessary
            if len(batch) < values_per_poly:
                batch = np.pad(batch, (0, values_per_poly - len(batch)))
            
            # First tier: Pack integers
            packed_ints = []
            for j in range(0, len(batch), self.int_batch_size):
                int_batch = batch[j:j + self.int_batch_size].tolist()
                if len(int_batch) < self.int_batch_size:
                    int_batch.extend([0] * (self.int_batch_size - len(int_batch)))
                packed_ints.append(self.int_encoder.encode(int_batch))
            
            # Second tier: Encode as polynomial
            poly = self.poly_encoder.encode(packed_ints)
            
            # Encrypt
            ct = self.rlwe.encrypt(poly)
            ciphertexts.append(ct)
            # FLOPs统计：多项式加密，粗略估算每个多项式加密约 2*N^2 乘法（N=poly_degree）
            self.flops_counter.encrypt += 2 * self.poly_degree * self.poly_degree
        
        return ciphertexts
    
    def aggregate_ciphertexts(self, ciphertext_list: List[List[Tuple[np.ndarray, np.ndarray]]]) -> List[Tuple[np.ndarray, np.ndarray]]:

        num_users = len(ciphertext_list)
        num_cts = len(ciphertext_list[0])
        
        aggregated = []
        
        for i in range(num_cts):
            # Start with first user's ciphertext
            agg_ct = ciphertext_list[0][i]
            
            # Add remaining users' ciphertexts
            for j in range(1, num_users):
                agg_ct = self.rlwe.add_ciphertexts(agg_ct, ciphertext_list[j][i])
                # FLOPs统计：多项式加法，粗略估算每个多项式加法约 N 次加法
                self.flops_counter.aggregate += self.poly_degree
            
            aggregated.append(agg_ct)
        
        return aggregated
    
    def decrypt_and_unpack(self, ciphertexts: List[Tuple[np.ndarray, np.ndarray]], 
                          original_size: int) -> np.ndarray:

        all_values = []
        
        for ct in ciphertexts:
            # Decrypt
            poly = self.rlwe.decrypt(ct)
            self.flops_counter.decrypt += 2 * self.poly_encoder.degree * self.poly_encoder.degree
            
            packed_ints = self.poly_encoder.decode(poly)
            
            for packed_int in packed_ints:
                if isinstance(packed_int, (int, np.integer)):
                    values = self.int_encoder.decode(int(packed_int))
                    all_values.extend(values)
        
        # Dequantize
        scale = 2**15
        gradients = np.array(all_values[:original_size], dtype=np.float64) / scale
        
        return gradients


def benchmark_twopack():
    
    print("=" * 60)
    print("TwoPack Benchmark")
    print("=" * 60)
    
    # Parameters
    num_users = 2
    gradient_size = 100000  # 100K parameters
    poly_degree = 4096
    int_batch_size = 4
    
    print(f"\nConfiguration:")
    print(f"  Number of users: {num_users}")
    print(f"  Gradient size: {gradient_size}")
    print(f"  Polynomial degree: {poly_degree}")
    print(f"  Integer batch size: {int_batch_size}")
    
    twopack = TwoPack(
        int_encoder_type='crt',
        poly_encoder_type='subr',
        int_batch_size=int_batch_size,
        poly_degree=poly_degree
    )
    
    print(f"\nLoading gradients from .npz files...")
    user_gradients = load_gradients_from_npz(num_users, gradient_size)
    
    print(f"\n{'='*60}")
    print("1. ENCRYPTION PHASE")
    print(f"{'='*60}")
    
    all_ciphertexts = []
    encrypt_times = []
    
    for i, gradients in enumerate(user_gradients):
        start = time.time()
        cts = twopack.pack_and_encrypt(gradients)
        encrypt_time = time.time() - start
        encrypt_times.append(encrypt_time)
        all_ciphertexts.append(cts)
        
        print(f"User {i+1}: {encrypt_time:.4f}s ({len(cts)} ciphertexts)")
    
    avg_encrypt_time = np.mean(encrypt_times)
    print(f"\nAverage encryption time per user: {avg_encrypt_time:.4f}s")
    

    
    start = time.time()
    aggregated_cts = twopack.aggregate_ciphertexts(all_ciphertexts)
    agg_time = time.time() - start
    
    print(f"Aggregation time: {agg_time:.4f}s")
    print(f"Time per ciphertext: {agg_time/len(aggregated_cts)*1000:.2f}ms")
    
    # Benchmark decryption
    
    start = time.time()
    recovered = twopack.decrypt_and_unpack(aggregated_cts, gradient_size)
    decrypt_time = time.time() - start
    
    print(f"Decryption time: {decrypt_time:.4f}s")
    
    
    # expected = sum(user_gradients)
    # error = np.max(np.abs(recovered - expected))
    # relative_error = error / (np.max(np.abs(expected)) + 1e-10)
    
    # print(f"Max absolute error: {error:.6e}")
    # print(f"Max relative error: {relative_error:.6e}")
    # print(f"Verification: {'PASSED' if relative_error < 0.01 else 'FAILED'}")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Average encryption time: {avg_encrypt_time:.4f}s")
    print(f"Aggregation time: {agg_time:.4f}s")
    print(f"Decryption time: {decrypt_time:.4f}s")
    print(f"Total time: {avg_encrypt_time * num_users + agg_time + decrypt_time:.4f}s")
    print(f"Ciphertext size: {len(aggregated_cts)} polynomials")
    print(f"Compression ratio: {gradient_size / len(aggregated_cts):.1f}x")
    twopack.flops_counter.report()


if __name__ == "__main__":
    np.random.seed(42)
    
    benchmark_twopack()
