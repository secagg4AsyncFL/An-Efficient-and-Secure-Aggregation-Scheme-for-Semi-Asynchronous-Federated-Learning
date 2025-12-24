import threading
import time
import numpy as np
from cryptography.hazmat.primitives.asymmetric import dh
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import serialization
import os
import struct
import concurrent.futures
import logging

class SecureAggregationSystem:
    def __init__(self, num_clients=3, num_rounds=1):
        self.num_clients = num_clients
        self.num_rounds = num_rounds
        self.parameters = dh.generate_parameters(generator=2, key_size=2048)
        self.server_private_key = self.parameters.generate_private_key()
        self.client_masks = {}
        self.client_public_keys = {}
        self.shared_keys = {}
        self.lock = threading.Lock()
        self.round_stats = []  # 存储每轮的统计数据
        
        # 设置日志记录
        logging.basicConfig(level=logging.INFO,
                          format='%(asctime)s - %(levelname)s - %(message)s',
                          datefmt='%Y-%m-%d %H:%M:%S.%f')
    
    def key_to_float(self, key_bytes):
        """将密钥转换为浮点数"""
        return struct.unpack('!f', key_bytes[:4])[0]
    
    def generate_new_mask(self, old_mask):
        """生成新的掩码"""
        start_time = time.time_ns()
        
        random_bytes = os.urandom(32)
        combined = old_mask + random_bytes
        digest = hashes.Hash(hashes.SHA256())
        digest.update(combined)
        new_mask = digest.finalize()
        
        end_time = time.time_ns()
        return new_mask, (end_time - start_time) / 1000
    
    def client_key_exchange(self, client_id):
        """客户端密钥协商"""
        start_time = time.time_ns()
        
        private_key = self.parameters.generate_private_key()
        public_key = private_key.public_key()
        
        with self.lock:
            self.client_public_keys[client_id] = public_key
            shared_key = private_key.exchange(self.server_private_key.public_key())
            
            derived_key = HKDF(
                algorithm=hashes.SHA256(),
                length=32,
                salt=None,
                info=b'handshake data',
            ).derive(shared_key)
            
            self.shared_keys[client_id] = derived_key
            
        end_time = time.time_ns()
        return (end_time - start_time) / 1000
    
    def encrypt_gradients(self, client_id, input_file, round_num):
        """加密梯度"""
        start_time = time.time_ns()
        
        gradients = np.load(input_file)
        mask_float = self.key_to_float(self.shared_keys[client_id])
        
        encrypted_arrays = {}
        for key in gradients.files:
            arr = gradients[key]
            encrypted_arrays[key] = arr + mask_float
            
        # output_file = f'client_{client_id}_round_{round_num}_encrypted.npz'
        output_file = f'./cnn/client_{client_id}_encrypted.npz'
        np.savez(output_file, **encrypted_arrays)
        
        end_time = time.time_ns()
        return (end_time - start_time) / 1000
    
    def aggregate_gradients(self, round_num):
        """聚合特定轮次的所有客户端加密梯度"""
        start_time = time.time_ns()
        
        # base_gradients = np.load(f'client_0_round_{round_num}_encrypted.npz')
        base_gradients = np.load(f'./cnn/client_0_encrypted.npz')
        aggregated = {key: base_gradients[key].copy() for key in base_gradients.files}
        
        for i in range(1, self.num_clients):
            # client_gradients = np.load(f'client_{i}_round_{round_num}_encrypted.npz')
            client_gradients = np.load(f'./cnn/client_{i}_encrypted.npz')
            for key in aggregated:
                aggregated[key] += client_gradients[key]
        
        # np.savez(f'aggregated_gradients_round_{round_num}.npz', **aggregated)
        np.savez(f'./cnn/aggregated_gradients.npz', **aggregated)
        end_time = time.time_ns()
        return (end_time - start_time) / 1000

    def update_all_masks(self):
        """更新所有客户端的掩码并测量时间"""
        start_time = time.time_ns()
        mask_update_times = []
        
        for client_id in self.shared_keys:
            new_mask, update_time = self.generate_new_mask(self.shared_keys[client_id])
            self.shared_keys[client_id] = new_mask
            mask_update_times.append(update_time)
            logging.info(f"客户端 {client_id} 完成掩码更新，耗时 {update_time:.2f} 微秒")
        
        end_time = time.time_ns()
        total_time = (end_time - start_time) / 1000
        
        return {
            'individual_times': mask_update_times,
            'total_time': total_time,
            'average_time': np.mean(mask_update_times)
        }

    def run_round(self, round_num, current_system_time):
        """运行单轮聚合"""
        logging.info(f"\n开始第 {round_num + 1} 轮聚合")
        round_start_time = time.time_ns()
        
        # 多线程进行梯度加密
        encryption_times = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_clients) as executor:
            future_to_client = {
                executor.submit(self.encrypt_gradients, i, './cnn/cnn.npz', round_num): i 
                for i in range(self.num_clients)
            }
            for future in concurrent.futures.as_completed(future_to_client):
                client_id = future_to_client[future]
                try:
                    time_taken = future.result()
                    encryption_times.append(time_taken)
                    logging.info(f"轮次 {round_num + 1}: 客户端 {client_id} 完成梯度加密，耗时 {time_taken:.2f} 微秒")
                except Exception as e:
                    logging.error(f"轮次 {round_num + 1}: 客户端 {client_id} 梯度加密失败: {str(e)}")
        
        # 执行梯度聚合
        aggregation_time = self.aggregate_gradients(round_num)
        logging.info(f"轮次 {round_num + 1}: 梯度聚合完成，耗时 {aggregation_time:.2f} 微秒")
        
        # 更新所有掩码并测量时间
        mask_update_stats = self.update_all_masks()
        logging.info(f"轮次 {round_num + 1}: 所有掩码更新完成，总耗时 {mask_update_stats['total_time']:.2f} 微秒")
        
        round_end_time = time.time_ns()
        round_total_time = (round_end_time - round_start_time) / 1000
        current_system_time = current_system_time + round_total_time
        # 返回本轮统计数据
        return {
            'round_number': round_num + 1,
            'encryption_total_time': sum(encryption_times),
            'encryption_avg_time': np.mean(encryption_times),
            'aggregation_time': aggregation_time,
            'mask_update_total_time': mask_update_stats['total_time'],
            'mask_update_avg_time': mask_update_stats['average_time'],
            'round_total_time': round_total_time,
            'current_system_time': current_system_time
        }

    def run_system(self):
        """运行整个系统"""
        system_start_time = time.time_ns()
        logging.info(f"开始运行安全聚合系统，总轮次: {self.num_rounds}")
        current_system_time = 0
        # 初始密钥协商
        key_exchange_times = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_clients) as executor:
            future_to_client = {
                executor.submit(self.client_key_exchange, i): i 
                for i in range(self.num_clients)
            }
            for future in concurrent.futures.as_completed(future_to_client):
                client_id = future_to_client[future]
                try:
                    time_taken = future.result()
                    key_exchange_times.append(time_taken)
                    logging.info(f"客户端 {client_id} 完成密钥协商，耗时 {time_taken:.2f} 微秒")
                except Exception as e:
                    logging.error(f"客户端 {client_id} 密钥协商失败: {str(e)}")
        
        # 运行多轮聚合
        for round_num in range(self.num_rounds):
            round_stats = self.run_round(round_num, current_system_time)
            self.round_stats.append(round_stats)
            current_system_time = round_stats['current_system_time']
        
        system_end_time = time.time_ns()
        total_system_time = (system_end_time - system_start_time) / 1000
        
        # 返回整体统计数据
        return {
            'key_exchange_total_time': sum(key_exchange_times),
            'key_exchange_avg_time': np.mean(key_exchange_times),
            'round_stats': self.round_stats,
            'total_system_time': total_system_time
        }

if __name__ == "__main__":
    # 创建系统实例并运行
    num_clients = 2  # 可以修改客户端数量
    num_rounds = 1   # 可以修改聚合轮次
    system = SecureAggregationSystem(num_clients, num_rounds)
    
    # 运行系统并获取性能统计
    stats = system.run_system()
    
    # 打印总体性能统计
    print("\n系统整体性能统计:")
    print(f"初始密钥协商总时间: {stats['key_exchange_total_time']:.2f} 微秒")
    print(f"初始密钥协商平均时间: {stats['key_exchange_avg_time']:.2f} 微秒")
    print(f"系统总运行时间: {stats['total_system_time']:.2f} 微秒")
    
    # 打印每轮统计
    print("\n各轮次性能统计:")
    for round_stat in stats['round_stats']:
        print(f"\n第 {round_stat['round_number']} 轮:")
        print(f"梯度加密总时间: {round_stat['encryption_total_time']:.2f} 微秒")
        print(f"梯度加密平均时间: {round_stat['encryption_avg_time']:.2f} 微秒")
        print(f"梯度聚合时间: {round_stat['aggregation_time']:.2f} 微秒")
        print(f"掩码更新总时间: {round_stat['mask_update_total_time']:.2f} 微秒")
        print(f"掩码更新平均时间: {round_stat['mask_update_avg_time']:.2f} 微秒")
        print(f"本轮总时间: {round_stat['round_total_time']:.2f} 微秒")
        print(f"当前总运行时间: {round_stat['current_system_time']:.2f} 微秒")
