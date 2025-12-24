"""
Secure Federated Learning with Buffered Asynchronous Aggregation (SecFedBuff)
Based on FedBuff with Diffie-Hellman key exchange and masked gradients
"""
from flgo.algorithm.asyncbase import AsyncServer
from flgo.algorithm.fedbase import BasicClient
import flgo.utils.fmodule as fmodule
import copy
import secrets
import hashlib
import random

# Diffie-Hellman parameters (these should be large primes in production)
DH_BASE = 2
DH_PRIME = 2**256 - 189  # A large prime for modulus
MASK_SCALE = 0.01  # Mask magnitude: 1% perturbation
DEBUG_MODE = True  # Set to True to disable masking for debugging

class Server(AsyncServer):
    def initialize(self):
        """Initialize the server with buffer parameters and DH setup"""
        super().initialize()
        self.buffer = []  # Buffer to store (gradient, round, client_id)
        self.client_masks = {}  # Mapping of client_id to mask for removing masks during aggregation
        self.eta = 0.5
        
        # Diffie-Hellman key setup
        self.private_key = secrets.randbelow(DH_PRIME - 1) + 1
        self.public_key = pow(DH_BASE, self.private_key, DH_PRIME)

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

        # Process each received update
        for i, (cdelta, ctau) in enumerate(zip(received_updates, received_client_taus)):
            client_id = received_client_ids[i] if i < len(received_client_ids) else i
            client_key = received_client_keys[i] if i < len(received_client_keys) else None
            
            # Compute shared key and generate mask
            if client_key is not None and isinstance(client_key, int) and client_key > 0:
                shared_key = self.compute_shared_key(client_key)
                mask = self.generate_mask_from_key(shared_key) if not DEBUG_MODE else 0.0
                
                # Store mask for removal during aggregation
                self.client_masks[client_id] = mask
                
                # Remove mask from the received update
                if mask != 0.0:
                    print(f"[DEBUG SERVER] Client {client_id}: Removing mask {mask}")
                    for param in cdelta.parameters():
                        param.data.sub_(mask)
            
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
        self.model = self.model + self.eta * model_delta
            
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

    def reply(self, svr_pkg):
        """Reply to server with masked gradient"""
        # Unpack model and perform training
        model = self.unpack(svr_pkg)
        global_model = copy.deepcopy(model)
        
        # Debug: check model before training
        model_param_before = list(model.parameters())[0].data.clone()
        
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
            'client_key': self.public_key
        })
        
        return cpkg