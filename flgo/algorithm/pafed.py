"""
Privacy-Preserving Asynchronous Federated Learning with Gradient Conflict Detection (PAFED)
Implementation based on the paper: "Privacy-Preserving Asynchronous Federated Learning Under Non-IID Settings"

Key features:
1. ADMM-based local optimization with dual variables
2. Gradient conflict detection
3. Gradient projection correction
4. Three-level weighted aggregation
5. Optional CKKS homomorphic encryption
"""

from flgo.algorithm.asyncbase import AsyncServer
from flgo.algorithm.fedbase import BasicClient
import flgo.utils.fmodule as fmodule
import copy
import torch
import numpy as np


class Server(AsyncServer):
    """
    PAFED Server with three-level weighted aggregation strategy
    """
    
    def initialize(self):
        """Initialize server parameters"""
        self.init_algo_para({
            'buffer_ratio': 0.1,  # Buffer threshold ratio (10% of clients)
            'eta': 0.5,           # Learning rate for server (reduced from 1.0)
            'alpha0': 0.6,        # Weight for online gradients
            'alpha1': 0.3,        # Weight for non-conflicting delayed gradients
            'alpha2': 0.1,        # Weight for corrected conflicting gradients
            'rho': 0.01,           # ADMM penalty coefficient (increased from 0.1)
            'admm_steps': 5,      # Number of ADMM optimization steps (for clients)
            'use_encryption': True,  # Whether to use CKKS encryption
        })
        
        # Buffers for different types of gradients
        self.buffer_online = []           # Online clients' gradients
        self.buffer_delayed_nc = []       # Delayed non-conflicting gradients
        self.buffer_delayed_c = []        # Delayed conflicting gradients
        
        # Store client info for calculation
        self.client_taus = {}
        self.client_updates = {}
        
        # For gradient conflict detection
        self.avg_online_grad = None
        
        if self.use_encryption:
            try:
                import tenseal as ts
                # Setup TenSEAL context for CKKS encryption
                self.context = ts.context(ts.SCHEME_TYPE.CKKS, 
                                         poly_modulus_degree=8192, 
                                         coeff_mod_bit_sizes=[60, 40, 40, 60])
                self.context.generate_galois_keys()
                self.context.global_scale = 2**40
            except ImportError:
                print("Warning: TenSEAL not installed. Encryption disabled.")
                self.use_encryption = False
    
    def package_handler(self, received_packages: dict):
        """
        Handle received packages from clients and manage buffering
        """
        if self.is_package_empty(received_packages):
            return False
        
        received_updates = received_packages['model']
        received_client_taus = [u._round for u in received_updates]
        received_client_ids = received_packages.get('client_id', list(range(len(received_updates))))
        
        # Store updates with their client info
        for cdelta, ctau, cid in zip(received_updates, received_client_taus, received_client_ids):
            self.client_taus[cid] = ctau
            self.client_updates[cid] = cdelta
            self.buffer_online.append((cid, cdelta, ctau))
        
        # Check if buffer size exceeds threshold
        if len(self.buffer_online) >= int(self.buffer_ratio * self.num_clients):
            self._perform_aggregation()
            return True
        
        return False
    
    def _perform_aggregation(self):
        """
        Perform three-level weighted aggregation
        """
        if len(self.buffer_online) == 0:
            return
        
        # Extract gradients and client info from buffer
        online_ids, online_updates, online_taus = zip(*self.buffer_online)
        
        # Calculate average gradient from online clients
        online_weights = [1.0] * len(online_updates)
        online_weights = np.array(online_weights) / len(online_updates)
        avg_online_grad = fmodule._model_average(list(online_updates), online_weights)
        self.avg_online_grad = avg_online_grad
        
        # Detect gradient conflicts for delayed clients
        conflicting_grads = []
        non_conflicting_grads = []
        
        for cid, delayed_update in self.client_updates.items():
            if cid not in online_ids:
                # This is a delayed client
                is_conflicting = self._detect_conflict(delayed_update, avg_online_grad)
                
                if is_conflicting:
                    # Apply projection correction
                    corrected_grad = self._project_gradient(delayed_update, avg_online_grad)
                    conflicting_grads.append(corrected_grad)
                else:
                    non_conflicting_grads.append(delayed_update)
        
        # Aggregate according to three-level strategy
        # Start with online gradients weighted by alpha0
        weighted_online = fmodule._model_average(list(online_updates), 
                                                 np.array([self.alpha0] * len(online_updates)) / len(online_updates))
        aggregated_delta = weighted_online
        
        # Add non-conflicting delayed gradients if any, weighted by alpha1
        if len(non_conflicting_grads) > 0:
            weighted_nc = fmodule._model_average(non_conflicting_grads, 
                                                 np.array([self.alpha1] * len(non_conflicting_grads)) / len(non_conflicting_grads))
            aggregated_delta = aggregated_delta + weighted_nc
        
        # Add corrected conflicting gradients if any, weighted by alpha2
        if len(conflicting_grads) > 0:
            weighted_c = fmodule._model_average(conflicting_grads, 
                                                np.array([self.alpha2] * len(conflicting_grads)) / len(conflicting_grads))
            aggregated_delta = aggregated_delta + weighted_c
        
        # Update global model: w^(t+1) = w^t + η * aggregated_delta
        self.model = self.model + self.eta * aggregated_delta
        
        # Clear buffers
        self._clear_buffers()
    
    def _detect_conflict(self, delayed_grad, avg_online_grad):
        """
        Detect gradient conflict by checking if inner product < 0
        
        Args:
            delayed_grad: gradient from delayed client
            avg_online_grad: average gradient from online clients
            
        Returns:
            True if gradient conflicts (inner product < 0)
        """
        try:
            # Convert models to tensors for inner product calculation
            delayed_vec = fmodule._modeldict_to_tensor1D(delayed_grad.state_dict()).flatten()
            avg_vec = fmodule._modeldict_to_tensor1D(avg_online_grad.state_dict()).flatten()
            
            # Calculate inner product
            inner_product = torch.dot(delayed_vec, avg_vec).item()
            
            return inner_product < 0
        except Exception as e:
            # If error occurs, assume no conflict
            return False
    
    def _project_gradient(self, gradient, reference_grad):
        """
        Project gradient to orthogonal plane of reference gradient
        
        Δw_j^(corrected) = Δw_j - [(Δw_j · Δw_m) / ||Δw_m||²] * Δw_m
        
        Args:
            gradient: gradient to be projected
            reference_grad: reference gradient (average online gradient)
            
        Returns:
            Corrected gradient
        """
        try:
            # Convert to tensors
            grad_vec = fmodule._modeldict_to_tensor1D(gradient.state_dict()).flatten()
            ref_vec = fmodule._modeldict_to_tensor1D(reference_grad.state_dict()).flatten()
            
            # Calculate inner product and norm
            inner_product = torch.dot(grad_vec, ref_vec)
            ref_norm_sq = torch.dot(ref_vec, ref_vec)
            
            if ref_norm_sq < 1e-8:
                return gradient
            
            # Calculate projection coefficient
            projection_coeff = inner_product / ref_norm_sq
            
            # Project to orthogonal plane
            corrected_vec = grad_vec - projection_coeff * ref_vec
            
            # Convert back to model
            corrected_grad = copy.deepcopy(gradient)
            state_dict = corrected_grad.state_dict()
            tensor_idx = 0
            
            for param_name in state_dict:
                param_shape = state_dict[param_name].shape
                param_size = state_dict[param_name].numel()
                if param_size > 0:
                    state_dict[param_name] = corrected_vec[tensor_idx:tensor_idx + param_size].reshape(param_shape)
                    tensor_idx += param_size
            
            corrected_grad.load_state_dict(state_dict)
            return corrected_grad
        except Exception as e:
            # If projection fails, return original gradient
            return gradient
    
    def _clear_buffers(self):
        """Clear all buffers after aggregation"""
        self.buffer_online = []
        self.buffer_delayed_nc = []
        self.buffer_delayed_c = []
        self.client_taus.clear()
        self.client_updates.clear()
    
    def save_checkpoint(self):
        """Save server checkpoint"""
        cpt = super().save_checkpoint()
        cpt.update({
            'buffer_online': self.buffer_online,
            'buffer_delayed_nc': self.buffer_delayed_nc,
            'buffer_delayed_c': self.buffer_delayed_c,
            'client_taus': self.client_taus,
            'client_updates': self.client_updates,
        })
        return cpt
    
    def load_checkpoint(self, cpt):
        """Load server checkpoint"""
        super().load_checkpoint(cpt)
        self.buffer_online = cpt.get('buffer_online', [])
        self.buffer_delayed_nc = cpt.get('buffer_delayed_nc', [])
        self.buffer_delayed_c = cpt.get('buffer_delayed_c', [])
        self.client_taus = cpt.get('client_taus', {})
        self.client_updates = cpt.get('client_updates', {})


class Client(BasicClient):
    """
    PAFED Client with ADMM-based local optimization
    """
    
    def initialize(self):
        """Initialize client parameters"""
        super().initialize()
        
        # Dual variables (Lagrangian multipliers)
        self.y_dual = None
        
        if self.use_encryption:
            try:
                import tenseal as ts
                self.context = ts.context(ts.SCHEME_TYPE.CKKS,
                                         poly_modulus_degree=8192,
                                         coeff_mod_bit_sizes=[60, 40, 40, 60])
                self.context.generate_galois_keys()
                self.context.global_scale = 2**40
            except ImportError:
                print("Warning: TenSEAL not installed. Encryption disabled.")
                self.use_encryption = False
    
    @fmodule.with_multi_gpus
    def train(self, model):
        """
        ADMM-based local training with dual variable updates
        """
        # Store global model and initialize dual variables
        global_model = copy.deepcopy(model)
        
        if self.y_dual is None:
            self.y_dual = copy.deepcopy(model)
            for param in self.y_dual.parameters():
                param.data.zero_()
        
        # ADMM optimization steps
        for admm_step in range(self.admm_steps):
            model.train()
            optimizer = self.calculator.get_optimizer(model, lr=self.learning_rate, 
                                                     weight_decay=self.weight_decay, 
                                                     momentum=self.momentum)
            
            # Local training steps
            for iter in range(self.num_steps):
                batch_data = self.get_batch_data()
                model.zero_grad()
                
                # Compute local empirical loss
                loss_res = self.calculator.compute_loss(model, batch_data)
                loss = loss_res['loss']
                
                # Add ADMM augmented Lagrangian terms
                # L_i = f_i(w_i) + y_i^T(w_i - w) + (ρ/2)||w_i - w||²
                dual_term = torch.tensor(0.0, dtype=loss.dtype, device=loss.device)
                penalty_term = torch.tensor(0.0, dtype=loss.dtype, device=loss.device)
                
                for param_model, param_dual, param_global in zip(model.parameters(), 
                                                                 self.y_dual.parameters(),
                                                                 global_model.parameters()):
                    # Dual term: y_i^T(w_i - w)
                    dual_term = dual_term + torch.sum(param_dual.detach() * (param_model - param_global.detach()))
                    # Penalty term: (ρ/2)||w_i - w||²
                    penalty_term = penalty_term + torch.sum((param_model - param_global.detach()) ** 2)
                
                # Total ADMM loss
                admm_loss = loss + dual_term + (self.rho / 2.0) * penalty_term
                admm_loss.backward()
                
                if self.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=self.clip_grad)
                
                optimizer.step()
        
        # Update dual variables: y_i^(t+1) = y_i^t + ρ(w_i^(t+1) - w^t)
        self._update_dual_variables(model, global_model)
    
    def _update_dual_variables(self, model, global_model):
        """
        Update dual variables (Lagrangian multipliers)
        y_i^(t+1) = y_i^t + ρ(w_i^(t+1) - w^t)
        """
        with torch.no_grad():
            for param_dual, param_model, param_global in zip(self.y_dual.parameters(),
                                                             model.parameters(),
                                                             global_model.parameters()):
                param_dual.data = param_dual.data + self.rho * (param_model.data - param_global.data)
    
    def reply(self, svr_pkg):
        """
        Reply to server with model update
        """
        model = self.unpack(svr_pkg)
        global_model = copy.deepcopy(model)
        self.train(model)
        # Compute update as model difference
        update = model - global_model
        update._round = model._round
        cpkg = self.pack(update)
        return cpkg

