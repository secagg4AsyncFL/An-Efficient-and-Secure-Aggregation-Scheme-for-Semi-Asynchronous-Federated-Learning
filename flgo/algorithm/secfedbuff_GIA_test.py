"""
Secure Federated Learning with Buffered Asynchronous Aggregation (SecFedBuff)
Based on FedBuff with Diffie-Hellman key exchange and masked gradients
With DLG/iDLG gradient inversion attack capabilities
"""
from flgo.algorithm.asyncbase import AsyncServer
from flgo.algorithm.fedbase import BasicClient
import flgo.utils.fmodule as fmodule
import copy
import secrets
import hashlib
import random
import torch
import torch.nn as nn
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from torchvision import transforms

# Diffie-Hellman parameters (these should be large primes in production)
DH_BASE = 2
DH_PRIME = 2**256 - 189  # A large prime for modulus
MASK_SCALE = 1.00  # Mask noise standard deviation (increased for effective protection)
DEBUG_MODE = False  # Set to True to disable masking for debugging (NO MASKING FOR GIA TESTING)

# GIA (Gradient Inversion Attack) parameters
GIA_ENABLED = True  # Enable gradient inversion attack
GIA_METHOD = 'iDLG'  # 'DLG' or 'iDLG'
GIA_ATTACK_ROUND = 1  # Which round to launch the attack (早期攻击效果更好)
GIA_NUM_ITERATIONS = 150  # Number of optimization iterations
GIA_LEARNING_RATE = 1.0  # Learning rate for LBFGS optimizer
GIA_SAVE_PATH = './gia_results'  # Path to save attack results
GIA_USE_SINGLE_STEP_GRAD = True  # 使用单步梯度而非模型更新（关键！）

class Server(AsyncServer):
    def initialize(self):
        """Initialize the server with buffer parameters and DH setup"""
        super().initialize()
        self.buffer = []  # Buffer to store (gradient, round, client_id)
        self.client_masks = {}  # Mapping of client_id to mask for removing masks during aggregation
        
        # Diffie-Hellman key setup
        self.private_key = secrets.randbelow(DH_PRIME - 1) + 1
        self.public_key = pow(DH_BASE, self.private_key, DH_PRIME)
        
        # GIA attack setup
        self.gia_enabled = self.option.get('gia_enabled', GIA_ENABLED)
        self.gia_method = self.option.get('gia_method', GIA_METHOD)
        self.gia_attack_round = self.option.get('gia_attack_round', GIA_ATTACK_ROUND)
        self.gia_num_iterations = self.option.get('gia_num_iterations', GIA_NUM_ITERATIONS)
        self.gia_learning_rate = self.option.get('gia_learning_rate', GIA_LEARNING_RATE)
        self.gia_save_path = self.option.get('gia_save_path', GIA_SAVE_PATH)
        self.gia_use_single_step_grad = self.option.get('gia_use_single_step_grad', GIA_USE_SINGLE_STEP_GRAD)
        self.gia_attacked = False  # Flag to track if attack has been performed
        
        # Create save directory for attack results
        if self.gia_enabled and not os.path.exists(self.gia_save_path):
            os.makedirs(self.gia_save_path)

    def compute_shared_key(self, client_public_key):
        """Compute shared key using client's public key"""
        return pow(client_public_key, self.private_key, DH_PRIME)

    def generate_mask_from_key(self, shared_key, scale=None):
        """
        Generate a controlled mask from shared key
        Args:
            shared_key: The shared key value
            scale: Controls the magnitude of the mask (default MASK_SCALE)
        Returns:
            A floating-point mask value
        """
        if scale is None:
            scale = MASK_SCALE
        # Use hash of the key as seed for random number generator
        key_hash = int(hashlib.sha256(str(shared_key).encode()).hexdigest(), 16)
        rng = random.Random(key_hash)
        # Generate mask in range [-scale, scale]
        return rng.uniform(-scale, scale)

    def package_handler(self, received_packages: dict):
        """Handle packages received from clients"""
        if self.is_package_empty(received_packages):
            return False

        received_updates = received_packages['model']
        received_client_taus = [u._round for u in received_updates]
        
        # Extract client IDs and keys - they should be lists from unpack
        received_client_ids = received_packages.get('client_id', list(range(len(received_updates))))
        received_client_keys = received_packages.get('client_key', [None] * len(received_updates))
        
        # Extract training data information for GIA attack
        received_client_data = received_packages.get('client_train_data', [None] * len(received_updates))
        # 获取单步梯度（如果可用）
        received_single_step_grads = received_packages.get('single_step_gradient', [None] * len(received_updates))
        # 获取原始训练数据（用于验证攻击效果）
        received_gt_data = received_packages.get('gt_data', [None] * len(received_updates))
        received_gt_label = received_packages.get('gt_label', [None] * len(received_updates))

        # Process each received update
        for i, (cdelta, ctau) in enumerate(zip(received_updates, received_client_taus)):
            client_id = received_client_ids[i] if i < len(received_client_ids) else i
            client_key = received_client_keys[i] if i < len(received_client_keys) else None
            client_data_info = received_client_data[i] if i < len(received_client_data) else None
            single_step_grad = received_single_step_grads[i] if i < len(received_single_step_grads) else None
            gt_data = received_gt_data[i] if i < len(received_gt_data) else None
            gt_label = received_gt_label[i] if i < len(received_gt_label) else None
            
            # Compute shared key and generate mask FIRST (before attack)
            # This ensures the attack uses the masked gradient
            mask_applied = 0.0
            if client_key is not None and isinstance(client_key, int) and client_key > 0:
                shared_key = self.compute_shared_key(client_key)
                mask_applied = self.generate_mask_from_key(shared_key) if not DEBUG_MODE else 0.0
                
                # Store mask for removal during aggregation
                self.client_masks[client_id] = mask_applied
            
            # Launch GIA attack BEFORE removing mask (to test attack on masked gradient)
            if (self.gia_enabled and not self.gia_attacked and 
                self.current_round >= self.gia_attack_round and client_data_info is not None):
                self.gv.logger.info(f"[GIA ATTACK] Launching {self.gia_method} attack at round {self.current_round} on client {client_id}")
                self.gv.logger.info(f"[GIA ATTACK] DEBUG_MODE={DEBUG_MODE}, Mask applied={mask_applied != 0.0}")
                
                # 关键修复：如果有掩码，对单步梯度也应用掩码
                if single_step_grad is not None and mask_applied != 0.0:
                    # 对单步梯度应用相同的掩码（模拟攻击者只能看到加掩码的梯度）
                    masked_single_step_grad = [g.clone() + mask_applied for g in single_step_grad]
                    gradient_to_use = masked_single_step_grad
                    self.gv.logger.info(f"[GIA ATTACK] Using MASKED single-step gradient (mask={mask_applied:.6f})")
                elif single_step_grad is not None:
                    gradient_to_use = single_step_grad
                    self.gv.logger.info("[GIA ATTACK] Using unmasked single-step gradient (DEBUG_MODE=True)")
                else:
                    gradient_to_use = cdelta  # 已经包含掩码
                    self.gv.logger.info("[GIA ATTACK] Using model update (with mask if applied)")
                
                self._launch_gia_attack(gradient_to_use, client_data_info, client_id, gt_data, gt_label)
                self.gia_attacked = True
            
            # Remove mask from the received update for aggregation
            if mask_applied != 0.0:
                print(f"[DEBUG SERVER] Client {client_id}: Removing mask {mask_applied}")
                for param in cdelta.parameters():
                    param.data.sub_(mask_applied)
            
            # Add to buffer with client_id
            self.buffer.append((cdelta, ctau, client_id))

        # Check if buffer is full and perform aggregation
        # In async setting, aggregate when buffer reaches threshold OR when we have at least 1 update
        self.buffer_ratio = 0.1
        buffer_threshold = max(int(self.buffer_ratio * self.num_clients), 1)
        
        if len(self.buffer) >= buffer_threshold:
            self._aggregate_buffer()
            return True
        
        return False

    def _aggregate_buffer(self):
        """Aggregate updates in the buffer and update the global model"""
        if not self.buffer:
            return
        
        # Extract updates and weights (consistent with FedBuff)
        updates_bf = [b[0] for b in self.buffer]
        taus_bf = [b[1] for b in self.buffer]
        client_ids_bf = [b[2] for b in self.buffer]
        
        # Calculate weights based on staleness
        weights_bf = [(1 + self.current_round - ctau) ** (-0.5) for ctau in taus_bf]
        
        # Normalize weights to sum to 1
        total_weight = sum(weights_bf)
        normalized_weights = [w / total_weight for w in weights_bf]
        
        # Perform weighted averaging with normalized weights
        model_delta = fmodule._model_average(updates_bf, normalized_weights)
        
        # Update global model
        self.model = self.model + self.learning_rate * model_delta
            
        # Clear buffer and masks
        self.buffer = []
        self.client_masks = {}

    def communicate_with(self, target_id, package={}):
        """
        Override to include DH parameters in the package sent to clients
        """
        # Create a copy to avoid modifying the original
        pkg = copy.deepcopy(package)
        pkg.update({
            'base': DH_BASE,
            'prime': DH_PRIME,
            'server_public_key': self.public_key
        })
        return super().communicate_with(target_id, pkg)

    def save_checkpoint(self):
        """Save checkpoint including buffer state"""
        cpt = super().save_checkpoint()
        cpt.update({
            'buffer': self.buffer,
            'client_masks': self.client_masks,
            'private_key': self.private_key,
            'public_key': self.public_key
        })
        return cpt

    def load_checkpoint(self, cpt):
        """Load checkpoint including buffer state"""
        super().load_checkpoint(cpt)
        self.buffer = cpt.get('buffer', [])
        self.client_masks = cpt.get('client_masks', {})
        self.private_key = cpt.get('private_key', self.private_key)
        self.public_key = cpt.get('public_key', self.public_key)

    def _launch_gia_attack(self, gradient_update, client_data_info, client_id, gt_data=None, gt_label_tensor=None):
        """
        Launch gradient inversion attack (DLG or iDLG)
        
        Args:
            gradient_update: The gradient from client (单步梯度列表或模型差值)
            client_data_info: Dictionary containing data shape and label info
            client_id: ID of the target client
            gt_data: Ground truth data for comparison (optional)
            gt_label_tensor: Ground truth label tensor for comparison (optional)
        """
        try:
            # Extract data information
            data_shape = client_data_info.get('data_shape')
            label_info = client_data_info.get('label')
            dataset_name = client_data_info.get('dataset', 'unknown')
            batch_size = client_data_info.get('batch_size', 1)
            num_classes = client_data_info.get('num_classes', 10)
            
            if data_shape is None:
                self.gv.logger.warning("[GIA ATTACK] Missing data shape information, attack aborted")
                return
            
            # Prepare gradient - 支持梯度列表或模型
            device = self.device
            if isinstance(gradient_update, list):
                # 已经是梯度列表（单步梯度）
                original_dy_dx = [g.clone().to(device) for g in gradient_update]
                self.gv.logger.info("[GIA ATTACK] Using single-step gradient (recommended)")
            else:
                # 是模型，提取参数作为梯度（注意：这不是真正的梯度！）
                self.gv.logger.warning("[GIA ATTACK] Warning: Using model update instead of single-step gradient, attack效果可能较差")
                original_dy_dx = [param.data.clone().to(device) for param in gradient_update.parameters()]
            
            # Get model architecture (use current global model)
            net = copy.deepcopy(self.model).to(device)
            net.eval()
            
            # Initialize dummy data
            dummy_data = torch.randn(data_shape).to(device).requires_grad_(True)
            criterion = nn.CrossEntropyLoss().to(device)
            
            # Setup optimizer and attack method
            if self.gia_method == 'DLG':
                # DLG: optimize both data and label
                dummy_label = torch.randn((batch_size, num_classes)).to(device).requires_grad_(True)
                optimizer = torch.optim.LBFGS([dummy_data, dummy_label], lr=self.gia_learning_rate)
                label_pred = None
            elif self.gia_method == 'iDLG':
                # iDLG: predict label from gradient, only optimize data
                optimizer = torch.optim.LBFGS([dummy_data], lr=self.gia_learning_rate)
                # Predict label from gradient (use the last FC layer gradient)
                try:
                    fc_grad = original_dy_dx[-2]  # 通常倒数第二个是最后FC层的权重
                    label_pred = torch.argmin(torch.sum(fc_grad, dim=-1), dim=-1).detach().reshape((batch_size,)).to(device)
                    self.gv.logger.info(f"[GIA ATTACK] iDLG predicted label: {label_pred.cpu().numpy()}")
                except Exception as e:
                    # Fallback: use provided label info if available
                    if label_info is not None:
                        label_pred = torch.tensor([label_info] * batch_size).long().to(device)
                        self.gv.logger.info(f"[GIA ATTACK] Using provided label: {label_pred.cpu().numpy()}")
                    else:
                        self.gv.logger.warning(f"[GIA ATTACK] Cannot predict label: {str(e)}")
                        return
            else:
                self.gv.logger.warning(f"[GIA ATTACK] Unknown method: {self.gia_method}")
                return
            
            # Attack loop
            losses = []
            mses = []
            history = []
            history_iters = []
            
            self.gv.logger.info(f"[GIA ATTACK] Starting {self.gia_method} optimization with {self.gia_num_iterations} iterations...")
            self.gv.logger.info(f"[GIA ATTACK] Data shape: {data_shape}, Batch size: {batch_size}")
            
            for iters in range(self.gia_num_iterations):
                def closure():
                    optimizer.zero_grad()
                    net.zero_grad()
                    pred = net(dummy_data)
                    
                    if self.gia_method == 'DLG':
                        dummy_loss = -torch.mean(torch.sum(
                            torch.softmax(dummy_label, -1) * torch.log(torch.softmax(pred, -1) + 1e-10), 
                            dim=-1))
                    else:  # iDLG
                        dummy_loss = criterion(pred, label_pred)
                    
                    dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)
                    
                    grad_diff = 0
                    for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                        grad_diff += ((gx - gy) ** 2).sum()
                    grad_diff.backward()
                    return grad_diff
                
                optimizer.step(closure)
                current_loss = closure().item()
                losses.append(current_loss)
                
                # 计算 MSE（如果有真实数据）
                if gt_data is not None:
                    mse = torch.mean((dummy_data - gt_data.to(device)) ** 2).item()
                    mses.append(mse)
                
                # Log progress
                if iters % max(1, self.gia_num_iterations // 10) == 0:
                    log_msg = f"[GIA ATTACK] Iter {iters}/{self.gia_num_iterations}, Loss: {current_loss:.8f}"
                    if mses:
                        log_msg += f", MSE: {mses[-1]:.8f}"
                    self.gv.logger.info(log_msg)
                    history.append(dummy_data.detach().cpu().clone())
                    history_iters.append(iters)
                
                # Early stopping
                if current_loss < 1e-6:
                    self.gv.logger.info(f"[GIA ATTACK] Converged at iteration {iters}")
                    break
            
            # Save results
            self._save_gia_results(
                dummy_data,
                losses,
                history,
                history_iters,
                client_id,
                data_shape,
                batch_size,
                gt_data,
                mses,
                dataset_name
            )
            
            # Report attack success
            final_loss = losses[-1]
            result_msg = f"[GIA ATTACK] Attack completed. Final loss: {final_loss:.8f}"
            if mses:
                result_msg += f", Final MSE: {mses[-1]:.8f}"
            self.gv.logger.info(result_msg)
            
        except Exception as e:
            self.gv.logger.error(f"[GIA ATTACK] Error during attack: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _save_gia_results(self, recovered_data, losses, history, history_iters, client_id, data_shape, batch_size=1, gt_data=None, mses=None, dataset_name='unknown'):
        """Save GIA attack results"""
        try:
            # Save recovered data
            save_file = os.path.join(
                self.gia_save_path,
                f'{self.gia_method}_{dataset_name}_round{self.current_round}_client{client_id}_batch{batch_size}.pt'
            )
            save_dict = {
                'recovered_data': recovered_data.detach().cpu(),
                'losses': losses,
                'data_shape': data_shape,
                'method': self.gia_method,
                'round': self.current_round,
                'client_id': client_id,
                'batch_size': batch_size,
                'dataset': dataset_name
            }
            if gt_data is not None:
                save_dict['gt_data'] = gt_data.detach().cpu()
            if mses:
                save_dict['mses'] = mses
            torch.save(save_dict, save_file)
            
            # Save visualization if image data
            if len(data_shape) == 4 and data_shape[1] in [1, 3]:  # Batch of images
                try:
                    tp = transforms.ToPILImage()
                    
                    # 创建对比图
                    # 只保留3张还原过程图像（首、中、末）
                    keep_num = 3
                    if len(history) >= keep_num:
                        select_indices = [0, len(history)//2, len(history)-1]
                    else:
                        select_indices = list(range(len(history)))
                    num_cols = len(select_indices)
                    fig, axes = plt.subplots(2, num_cols, figsize=(3*num_cols, 6))
                    if num_cols == 1:
                        axes = axes.reshape(2, 1)
                    # 第一行：恢复过程
                    for plot_idx, hist_idx in enumerate(select_indices):
                        hist_data = history[hist_idx]
                        hist_iter = history_iters[hist_idx]
                        ax = axes[0, plot_idx]
                        img_data = hist_data[0].clamp(0, 1)
                        if data_shape[1] == 1:  # Grayscale
                            ax.imshow(img_data.squeeze().numpy(), cmap='gray')
                        else:  # RGB
                            ax.imshow(tp(img_data))
                        ax.set_title(f'Iter {hist_iter}')
                        ax.axis('off')
                    # 隐藏未使用的子图
                    for idx in range(num_cols, axes[0].shape[0]):
                        axes[0, idx].axis('off')
                    
                    # 第二行：最终结果和真实图像对比
                    ax = axes[1, 0]
                    final_img = recovered_data[0].detach().cpu().clamp(0, 1)
                    if data_shape[1] == 1:
                        ax.imshow(final_img.squeeze().numpy(), cmap='gray')
                    else:
                        ax.imshow(tp(final_img))
                    ax.set_title('Final Recovered')
                    ax.axis('off')
                    
                    if gt_data is not None:
                        ax = axes[1, 1]
                        gt_img = gt_data[0].detach().cpu().clamp(0, 1)
                        if data_shape[1] == 1:
                            ax.imshow(gt_img.squeeze().numpy(), cmap='gray')
                        else:
                            ax.imshow(tp(gt_img))
                        ax.set_title('Ground Truth')
                        ax.axis('off')
                        start_idx = 2
                    else:
                        start_idx = 1
                    
                    # 隐藏剩余子图
                    for idx in range(start_idx, num_cols):
                        axes[1, idx].axis('off')
                    
                    title = f'{self.gia_method} Attack - Round {self.current_round} - Client {client_id}\nFinal Loss: {losses[-1]:.6f}'
                    if mses:
                        title += f', MSE: {mses[-1]:.6f}'
                    plt.suptitle(title)
                    plt.tight_layout()
                    img_file = os.path.join(
                        self.gia_save_path,
                        f'{self.gia_method}_{dataset_name}_round{self.current_round}_client{client_id}_batch{batch_size}.png'
                    )
                    plt.savefig(img_file, dpi=150)
                    plt.close()
                    self.gv.logger.info(f"[GIA ATTACK] Visualization saved to {img_file}")
                except Exception as e:
                    self.gv.logger.warning(f"[GIA ATTACK] Could not save visualization: {str(e)}")
            
            self.gv.logger.info(f"[GIA ATTACK] Results saved to {save_file}")
            
        except Exception as e:
            self.gv.logger.error(f"[GIA ATTACK] Error saving results: {str(e)}")


class Client(BasicClient):
    def __init__(self, option):
        super().__init__(option)
        # Diffie-Hellman key setup
        self.private_key = secrets.randbelow(DH_PRIME - 1) + 1
        self.public_key = None
        self.shared_key = None
        self.mask = None

    def generate_public_key(self, base, prime):
        """Generate client's public key"""
        self.public_key = pow(base, self.private_key, prime)

    def compute_shared_key(self, server_public_key, prime):
        """Compute shared key using server's public key"""
        return pow(server_public_key, self.private_key, prime)

    def generate_mask_from_key(self, shared_key, scale=None):
        """
        Generate a controlled mask from shared key
        Args:
            shared_key: The shared key value
            scale: Controls the magnitude of the mask (default MASK_SCALE)
        Returns:
            A floating-point mask value
        """
        if scale is None:
            scale = MASK_SCALE
        # Use hash of the key as seed for random number generator
        key_hash = int(hashlib.sha256(str(shared_key).encode()).hexdigest(), 16)
        rng = random.Random(key_hash)
        # Generate mask in range [-scale, scale]
        return rng.uniform(-scale, scale)

    def _infer_dataset_name(self):
        """Infer dataset name from task path"""
        task_path = self.option.get('task', '') or ''
        if task_path == '':
            return 'unknown'
        parts = task_path.replace('\\', '/').split('/')
        for part in parts:
            lower = part.lower()
            if (part.startswith('test_') or 'mnist' in lower or 'cifar' in lower or
                    'fashion' in lower or 'emnist' in lower or 'svhn' in lower):
                return part
        return parts[-1] if parts else 'unknown'

    def reply(self, svr_pkg):
        """Reply to server with masked gradient"""
        # Unpack model and perform training
        model = self.unpack(svr_pkg)
        global_model = copy.deepcopy(model)
        
        # Debug: check model before training
        model_param_before = list(model.parameters())[0].data.clone()
        
        # Get training data info before training (for GIA attack)
        train_data_info = None
        single_step_gradient = None
        gt_data = None
        gt_label = None
        dataset_name = self._infer_dataset_name()
        
        if hasattr(self, 'train_data') and self.train_data is not None:
            try:
                # Get a sample batch to extract shape information
                batch_data = self.get_batch_data()
                if batch_data is not None and len(batch_data) >= 2:
                    data_sample, label_sample = batch_data[0], batch_data[1]
                    train_data_info = {
                        'data_shape': list(data_sample.shape),
                        'label': label_sample[0].item() if torch.is_tensor(label_sample) else label_sample,
                        'batch_size': len(label_sample),
                        'num_classes': self.option.get('num_classes', 10),
                        'dataset': dataset_name
                    }
                    
                    # 保存真实数据用于验证攻击效果
                    gt_data = data_sample.clone()
                    gt_label = label_sample.clone()
                    
                    # 计算单步梯度（这是 DLG/iDLG 真正需要的！）
                    if self.option.get('gia_use_single_step_grad', True):
                        try:
                            device = model.get_device() if hasattr(model, 'get_device') else next(model.parameters()).device
                            attack_model = copy.deepcopy(global_model).to(device)
                            attack_model.train()
                            
                            data_dev = data_sample.to(device)
                            label_dev = label_sample.to(device)
                            
                            criterion = nn.CrossEntropyLoss()
                            attack_model.zero_grad()
                            output = attack_model(data_dev)
                            loss = criterion(output, label_dev)
                            loss.backward()
                            
                            # 提取梯度
                            single_step_gradient = [param.grad.detach().clone() for param in attack_model.parameters()]
                        except Exception as e:
                            pass  # 如果计算单步梯度失败，仍然使用模型更新
            except Exception as e:
                pass  # Silently fail if we can't get training data info
        
        self.train(model)
        update = model - global_model
        update._round = model._round
        
        # Debug: check gradient
        model_param_after = list(model.parameters())[0].data.clone()
        update_param = list(update.parameters())[0].data.clone()

        # Extract DH parameters from server package
        base = svr_pkg.get('base', DH_BASE)
        prime = svr_pkg.get('prime', DH_PRIME)
        server_public_key = svr_pkg.get('server_public_key')
        

        # Perform DH key exchange
        if server_public_key is not None:
            self.generate_public_key(base, prime)
            self.shared_key = self.compute_shared_key(server_public_key, prime)
            
            # Generate mask from shared key
            self.mask = self.generate_mask_from_key(self.shared_key) if not DEBUG_MODE else 0.0
            
            # Add mask to gradient parameters
            if self.mask != 0.0:
                for param in update.parameters():
                    param.data.add_(self.mask)

        # Pack update and add client information
        cpkg = self.pack(update)
        cpkg.update({
            'client_id': self.client_id,
            'client_key': self.public_key,
            'client_train_data': train_data_info,  # Include training data info for GIA
            'single_step_gradient': single_step_gradient,  # 单步梯度（攻击用）
            'gt_data': gt_data,  # 真实数据（验证用）
            'gt_label': gt_label  # 真实标签（验证用）
        })
        
        return cpkg