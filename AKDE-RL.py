#!/usr/bin/env python
# coding: utf-8


import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, mean_squared_error
from scipy.stats import wasserstein_distance
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings('ignore')


TASK_TYPE = 'regression'    # Task type: choose between 'classification' or 'regression'
DATA_PATH = '.csv'        # Path to the data file
FEATURE_COLS = [    ]  # Names or indices of feature columns
LABEL_COL = ''           # Name or index of the target label column
SAVE_DIR = ''
RESULT_PREFIX = 'best_result'   # Prefix for the output result files


# --------------- Training Parameters ---------------
EPISODES = 100                   # Number of training episodes
BATCH_SIZE = 144                 # Batch size
LEARNING_RATE = 1e-4             # Learning rate
GAMMA = 0.95                     # Discount factor (for reward discounting)
HIDDEN_DIM = 256                 # Dimension of hidden layers



# ================ MC & AKDE Parameters (New) ================
MC_ROLLOUTS = 5          # Number of Monte Carlo rollouts per action (multi-step exploration samples)
AKDE_ENABLED = True      # Enable AKDE validation (accept/reject)
AKDE_BANDWIDTH = 0.28    # KDE bandwidth (value from paper)
AKDE_ALPHA = 0.05        # Threshold based on the alpha percentile of training set log-density
MIN_ACCEPT_RATIO = 0.01  # If acceptance rate is lower than this, retain top-k (proportion)



# ===================== Configuration Constraints =====================
DATA_CONSTRAINTS = {
    0: (-10, None, 'float'), 1: (-10, None, 'float'), 2: (-10, None, 'float'),
    3: (-10, None, 'float'), 4: (-10, None, 'float'), 5: (-10, None, 'float'),
    6: (-10, None, 'float'), 7: (-10, None, 'float'), 8: (-10, None, 'float'),
    9: (-10, None, 'float'), 10: (-10, None, 'float'), 11: (-10, None, 'float'),
    12: (-10, None, 'float'), 13: (-10, None, 'float')
}

LABEL_CONSTRAINTS = {
    'min': -10,
    'max': None,
    'dtype': 'float'  # Data type
}


# ===================== Data Loading Module =====================
def load_data():
    """Safely load data and validate integrity"""
    try:
        # 1. Check if the file exists
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(f"Data file not found: {DATA_PATH}")
        
        # 2. Read CSV file
        df = pd.read_csv(DATA_PATH)
        print("Successfully loaded data. Columns:", df.columns.tolist())
        
        # --- Process Features (X) ---
        # Handle both string column names and integer indices
        if isinstance(FEATURE_COLS[0], str):
            # Case A: FEATURE_COLS contains column names (e.g., ['col_a', 'col_b'])
            missing = [col for col in FEATURE_COLS if col not in df.columns]
            if missing:
                raise KeyError(f"Missing feature columns: {missing}")
            X = df[FEATURE_COLS].values.astype(np.float32)
            feature_names = FEATURE_COLS
        else:
            # Case B: FEATURE_COLS contains integer indices (e.g., [0, 1, 2])
            feature_names = df.columns[FEATURE_COLS].tolist()
            X = df.iloc[:, FEATURE_COLS].values.astype(np.float32)
            
        # --- Process Label (y) ---
        # Handle both string column name and integer index
        if isinstance(LABEL_COL, str):
            # Case A: LABEL_COL is a column name
            if LABEL_COL not in df.columns:
                raise KeyError(f"Label column not found: {LABEL_COL}")
            y = df[LABEL_COL].values
        else:
            # Case B: LABEL_COL is an integer index
            y = df.iloc[:, LABEL_COL].values
            
        # Ensure y is a 1D array (flatten)
        y = y.reshape(-1)
        
        # 3. Verify shapes and return
        print(f"Data shape: X={X.shape}, y={y.shape}")
        return X, y, feature_names
        
    except Exception as e:
        # Catch any error during loading and exit
        print(f"\n!! Data loading error: {str(e)} !!")
        exit(1)


# ===================== Reinforcement Learning Environment =====================
class RLEnvironment:
    def __init__(self, X, y):
        self.X_orig = X
        self.y_orig = y
        
        # Split data into training and validation sets (80/20 split)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Initialize baseline model and performance
        self.model = self._init_model()
        self.baseline = self._train_baseline()

        # --- AKDE Initialization ---
        self.kde = None
        self.kde_threshold = None
        if AKDE_ENABLED:
            try:
                # Fit KDE on training data to estimate the underlying probability density
                self.kde = KernelDensity(bandwidth=AKDE_BANDWIDTH)
                self.kde.fit(self.X_train)
                
                # Calculate threshold based on alpha percentile of log-density
                # Samples with density lower than this will be rejected
                train_scores = self.kde.score_samples(self.X_train)
                perc = np.percentile(train_scores, AKDE_ALPHA * 100)
                self.kde_threshold = float(perc)
            except Exception as e:
                print("AKDE initialization failed, disabling AKDE. Error:", e)
                self.kde = None
                self.kde_threshold = None

        # High exposure threshold (for diagnostic purposes, e.g., top 5% labels)
        try:
            self.high_exposure_threshold = float(np.percentile(self.y_orig, 95))
        except Exception:
            self.high_exposure_threshold = None

    def _init_model(self):
        # Initialize model based on task type (Classification vs Regression)
        return RandomForestClassifier(n_estimators=50) if TASK_TYPE == 'classification' \
            else RandomForestRegressor(n_estimators=50)

    def _train_baseline(self):
        # Train the initial model on original data and evaluate baseline performance
        self.model.fit(self.X_train, self.y_train)
        return self._evaluate()

    def _evaluate(self):
        # Evaluate current model performance on the validation set
        pred = self.model.predict(self.X_val)
        if TASK_TYPE == 'classification':
            return accuracy_score(self.y_val, pred)
        # For regression, return negative MSE (since we want to maximize reward)
        return -mean_squared_error(self.y_val, pred)

    def _akde_filter(self, X_generated):
        """
        Use fitted KDE for accept/reject sampling to ensure generated data 
        lies within the high-density region of the training distribution.
        
        Returns: accepted_samples, acceptance_rate
        """
        if (self.kde is None) or (self.kde_threshold is None):
            return X_generated, 1.0
            
        try:
            # Calculate log-density scores for generated samples
            gen_scores = self.kde.score_samples(X_generated)
            accepted_mask = gen_scores >= self.kde_threshold
            acc_rate = float(np.mean(accepted_mask))
            
            # If acceptance rate is too low, retain top-k highest density samples
            # to prevent total rejection while maintaining quality
            if acc_rate < MIN_ACCEPT_RATIO:
                k = max(1, int(len(X_generated) * MIN_ACCEPT_RATIO))
                top_idx = np.argsort(gen_scores)[-k:] # Get indices of top k scores
                accepted = X_generated[top_idx]
                return accepted, float(len(accepted) / len(X_generated))
                
            accepted = X_generated[accepted_mask]
            return accepted, acc_rate
        except Exception:
            return np.zeros((0, X_generated.shape[1])), 0.0

    def simulate_enhancement(self, X_new, y_new):
        """
        Simulate enhancement using a TEMPORARY model to avoid polluting the main environment.
        This is crucial for Monte Carlo rollouts where we need to test "what-if" scenarios.
        """
        # 1. AKDE Filtering: Filter out unrealistic samples
        X_accepted, acc_rate = self._akde_filter(X_new)
        
        if X_accepted.shape[0] == 0:
            # Heavy penalty if no valid samples are generated
            w_dist = 1e6
            reward_value = (self._evaluate() - self.baseline) - 0.5 * w_dist
            return reward_value
            
        # 2. Map accepted samples back to their corresponding labels (y)
        # Strategy: Exact match by bytes first; fallback to nearest neighbor if failed
        gen_bytes = [row.tobytes() for row in X_new]
        bytes_to_idx = {}
        for idx, b in enumerate(gen_bytes):
            if b not in bytes_to_idx:
                bytes_to_idx[b] = idx
                
        accepted_indices = []
        for row in X_accepted:
            b = row.tobytes()
            if b in bytes_to_idx:
                accepted_indices.append(bytes_to_idx[b])
            else:
                # Fallback: find closest sample in original list (should rarely happen)
                dists = np.linalg.norm(X_new - row, axis=1)
                accepted_indices.append(int(np.argmin(dists)))
                
        accepted_indices = np.array(accepted_indices, dtype=int)
        y_accepted = y_new[accepted_indices]

        # 3. Train temporary model and evaluate performance gain
        if TASK_TYPE == 'classification':
            tmp_model = RandomForestClassifier(n_estimators=50)
        else:
            tmp_model = RandomForestRegressor(n_estimators=50)
            
        # Combine original training data with accepted generated data
        X_combined = np.vstack([self.X_train, X_accepted])
        y_combined = np.concatenate([self.y_train, y_accepted])
        tmp_model.fit(X_combined, y_combined)
        
        # Evaluate on validation set
        pred = tmp_model.predict(self.X_val)
        if TASK_TYPE == 'classification':
            current_score = accuracy_score(self.y_val, pred)
        else:
            current_score = -mean_squared_error(self.y_val, pred)
            
        delta_perf = current_score - self.baseline
        
        # 4. Calculate distribution shift penalty (Wasserstein distance)
        # We want to minimize the distance between original and generated distributions
        try:
            w_dist = np.mean([wasserstein_distance(self.X_train[:, i], X_accepted[:, i])
                              for i in range(self.X_train.shape[1])])
        except Exception:
            w_dist = float('inf')
            
        # Final Reward: Performance Gain - Regularization Term
        reward_value = delta_perf - 0.5 * w_dist
        return reward_value

    def evaluate_enhancement(self, X_new, y_new):
        """
        Original evaluation function: updates the MAIN self.model.
        Note: We use simulate_enhancement for Monte Carlo to keep the main environment clean,
        but this function applies the change permanently.
        """
        # 1. AKDE Filtering
        X_accepted, acc_rate = self._akde_filter(X_new)
        if X_accepted.shape[0] == 0:
            w_dist = 1e6
            reward_value = (self._evaluate() - self.baseline) - 0.5 * w_dist
            return reward_value
            
        # 2. Map accepted samples to labels (same logic as simulate_enhancement)
        gen_bytes = [row.tobytes() for row in X_new]
        bytes_to_idx = {}
        for idx, b in enumerate(gen_bytes):
            if b not in bytes_to_idx:
                bytes_to_idx[b] = idx
        accepted_indices = []
        for row in X_accepted:
            b = row.tobytes()
            if b in bytes_to_idx:
                accepted_indices.append(bytes_to_idx[b])
            else:
                dists = np.linalg.norm(X_new - row, axis=1)
                accepted_indices.append(int(np.argmin(dists)))
        accepted_indices = np.array(accepted_indices, dtype=int)
        y_accepted = y_new[accepted_indices]

        # 3. Merge data and retrain the MAIN model (this modifies self.model state)
        X_combined = np.vstack([self.X_train, X_accepted])
        y_combined = np.concatenate([self.y_train, y_accepted])
        self.model.fit(X_combined, y_combined)
        
        current_score = self._evaluate()
        delta_perf = current_score - self.baseline
        
        # 4. Calculate Wasserstein distance penalty
        try:
            w_dist = np.mean([wasserstein_distance(self.X_train[:, i], X_accepted[:, i])
                              for i in range(self.X_train.shape[1])])
        except Exception:
            w_dist = float('inf')
            
        reward_value = delta_perf - 0.5 * w_dist
        return reward_value




# ===================== Actor-Critic Network =====================
class ActorCritic(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # Shared feature extraction layers
        # Translates input state to a hidden representation
        self.shared = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIM),
            nn.ReLU()
        )
        
        # Actor head: outputs 2 raw action values (unbounded means for the Gaussian distribution)
        self.actor = nn.Linear(HIDDEN_DIM, 2)
        
        # Critic head: outputs the estimated state value V(s)
        self.critic = nn.Linear(HIDDEN_DIM, 1)
        
        # Log standard deviation parameter for the Gaussian policy
        # This is a learnable parameter, allowing the agent to adjust its exploration noise
        self.log_std = nn.Parameter(torch.zeros(2))

    def forward(self, x):
        # Pass input through shared layers
        h = self.shared(x)
        
        # Get action means from the actor head
        means = self.actor(h)
        
        # Get state value from the critic head and remove the last dimension (squeeze)
        value = self.critic(h).squeeze(-1)
        
        # Calculate standard deviation from log_std
        # Clamping ensures std is never zero or negative to prevent numerical instability
        std = torch.exp(self.log_std).clamp(min=1e-6)
        
        return means, std, value

# ===================== Agent (Actor-Critic + MC rollouts) =====================
class AugmentAgent:
    def __init__(self, input_dim):
        # Initialize the network
        self.net = ActorCritic(input_dim)
        # Initialize the Adam optimizer
        self.optimizer = optim.Adam(self.net.parameters(), lr=LEARNING_RATE)

    def get_action_and_value(self, states):
        """
        Input: states (numpy array of shape [batch, dim])
        Returns: actions (numpy array [batch, 2]), log_probs (torch tensor), values (torch tensor)
        """
        # Convert numpy states to PyTorch tensor
        states_t = torch.FloatTensor(states)
        
        # Forward pass to get distribution parameters and value
        means, stds, values = self.net(states_t)
        
        # Create a Normal distribution for sampling actions
        dists = torch.distributions.Normal(means, stds)
        
        # Sample actions
        actions = dists.sample()
        
        # Calculate log probability of the sampled actions (sum across action dimensions)
        log_probs = dists.log_prob(actions).sum(dim=1)
        
        # Return actions as numpy for environment interaction, keep tensors for training
        return actions.detach().numpy(), log_probs, values.detach()

    def update(self, states, actions, returns, log_probs, values, entropy_coef=1e-3, value_coef=0.5):
        """
        Updates the Actor-Critic network using Advantage estimation.
        
        Logic:
          advantage = returns - values
          actor_loss = - E[log_prob * advantage]  (Policy Gradient)
          critic_loss = MSE(returns, values)      (Value Function Regression)
          Adds an entropy bonus to encourage exploration.
        """
        # Convert data to PyTorch tensors
        states_t = torch.FloatTensor(states)
        actions_t = torch.FloatTensor(actions)
        returns_t = torch.FloatTensor(returns)
        log_probs_t = log_probs
        values_t = values

        # Calculate Advantage: How much better was the outcome than expected?
        # .detach() prevents gradients from flowing back into the value network during actor update
        advantages = returns_t - values_t.detach()
        
        # Actor Loss: Maximize log probability of actions that led to positive advantage
        actor_loss = -(log_probs_t * advantages).mean()
        
        # Critic Loss: Minimize the squared error between predicted value and actual return
        critic_loss = (returns_t - values_t).pow(2).mean()
        
        # Entropy Calculation: Measure of randomness in the policy
        # We want to maximize entropy (minimize negative entropy) to encourage exploration
        means, stds, _ = self.net(states_t)
        dist = torch.distributions.Normal(means, stds)
        entropy = dist.entropy().sum(dim=1).mean()

        # Total Loss
        # Note: We subtract entropy because we want to maximize it (equivalent to minimizing -entropy)
        loss = actor_loss + value_coef * critic_loss - entropy_coef * entropy

        # Optimization step
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        
        self.optimizer.step()



# ===================== Data Generator (Handles single actions) =====================

class DataValidator:
    @staticmethod
    def apply_constraints(data, constraints):
        """
        Apply min/max clipping and type casting to data columns based on constraints.
        
        Args:
            data: numpy array of shape (batch, dim) or (1, dim)
            constraints: dictionary mapping column index to (min, max, dtype)
        """
        constrained_data = data.copy()
        
        # Iterate through each constraint definition
        for col_idx, (min_val, max_val, dtype) in constraints.items():
            # Clip values to the specified range [min_val, max_val]
            constrained_data[:, col_idx] = np.clip(constrained_data[:, col_idx], min_val, max_val)
            
            # Cast to the required data type
            if dtype == 'int':
                constrained_data[:, col_idx] = np.round(constrained_data[:, col_idx]).astype(int)
            elif dtype == 'float':
                constrained_data[:, col_idx] = constrained_data[:, col_idx].astype(float)
                
        return constrained_data

class DataGenerator:
    @staticmethod
    def generate_batch(X, y, actions, feature_constraints=None, label_constraints=None):
        """
        Generate augmented data batch based on action parameters.
        
        Action Mapping Logic:
          - alpha = sigmoid(a0): Maps action[0] to range [0, 1] for interpolation weight.
          - beta  = 0.1 * (tanh(a1) + 1): Maps action[1] to range [0, 0.2] for noise scale.
        
        Returns:
            X_new: Augmented features (batch, dim)
            y_new: Corresponding labels (batch,)
        """
        # Ensure actions is 2D (batch_size, action_dim)
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)
            
        # Calculate interpolation weight alpha using sigmoid function
        alphas = 1.0 / (1.0 + np.exp(-actions[:, 0]))  # Range: (0, 1)
        
        # Calculate noise magnitude beta using tanh function
        betas = 0.1 * (np.tanh(actions[:, 1]) + 1.0)    # Range: (0, 0.2)
        
        X_new = []
        y_new = []
        
        # Randomly select pairs of indices from original dataset for interpolation
        # Shape: (batch_size, 2)
        indices = np.random.choice(len(X), (len(actions), 2), replace=True)
        
        for i in range(len(actions)):
            alpha = float(alphas[i])
            beta = float(betas[i])
            idx1, idx2 = indices[i]
            
            # --- Generate New Feature Sample ---
            # Formula: Mix two samples + Add Gaussian noise
            sample = alpha * X[idx1] + (1 - alpha) * X[idx2] + beta * np.random.randn(*X[0].shape)
            
            # Apply feature constraints (clipping & typing)
            if feature_constraints:
                sample = DataValidator.apply_constraints(sample.reshape(1, -1), feature_constraints).flatten()
                
            X_new.append(sample)
            
            # --- Generate New Label ---
            if TASK_TYPE == 'classification':
                # For classification: randomly pick one of the parent labels (majority vote style)
                label = y[idx1] if np.random.rand() > 0.5 else y[idx2]
            else:
                # For regression: interpolate labels using the same alpha weight
                label = alpha * y[idx1] + (1 - alpha) * y[idx2]
                
            # Apply label constraints
            if label_constraints:
                label = np.clip(label, label_constraints.get('min', -np.inf), label_constraints.get('max', np.inf))
                dtype = label_constraints.get('dtype', 'float')
                
                if dtype == 'int':
                    label = int(round(label))
                elif dtype == 'float':
                    label = float(label)
                    
            y_new.append(label)
            
        return np.array(X_new), np.array(y_new)




# ===================== Visualization & Saving Module =====================

class ResultVisualizer:
    @staticmethod
    def plot_pca_comparison(original, augmented, save_path):
        """
        Visualizes the distribution of original vs. augmented data using PCA.
        Fits PCA on combined data to ensure consistent projection space.
        """
        # Initialize PCA to reduce dimensions to 2D for visualization
        pca = PCA(n_components=2)
        
        # Combine datasets to fit PCA on the global structure
        combined = np.vstack([original, augmented])
        projected = pca.fit_transform(combined)
        
        plt.figure(figsize=(10, 6))
        
        # Plot Original Data (First N points)
        plt.scatter(projected[:len(original), 0], projected[:len(original), 1], 
                    alpha=0.3, label='Original Data')
                    
        # Plot Augmented Data (Remaining points)
        plt.scatter(projected[len(original):, 0], projected[len(original):, 1], 
                    alpha=0.3, label='Augmented Data')
        
        # Title includes total explained variance ratio
        plt.title(f"PCA Projection Comparison (Explained Variance: {pca.explained_variance_ratio_.sum():.2f})")
        plt.legend()
        
        # Save with high DPI for clarity
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    @staticmethod
    def save_training_history(history, save_path):
        """
        Plots training progress: Reward trends and Distribution Distance (Wasserstein).
        """
        plt.figure(figsize=(15, 6))
        
        # --- Subplot 1: Reward Curve ---
        plt.subplot(1, 2, 1)
        plt.plot(history['episode'], history['reward'], color='#2e7d32', linewidth=2, label='Total Reward')
        plt.title('Reward Trend', fontsize=14, pad=20)
        plt.xlabel('Episode', fontsize=12)
        plt.ylabel('Reward Value', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # --- Subplot 2: Wasserstein Distance Curve ---
        plt.subplot(1, 2, 2)
        plt.plot(history['episode'], history['w_dist'], color='#c62828', linewidth=2, label='Wasserstein Distance')
        plt.title('Data Distribution Distance', fontsize=14, pad=20)
        plt.xlabel('Episode', fontsize=12)
        plt.ylabel('W-Distance', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Adjust layout to prevent overlap
        plt.tight_layout(pad=3.0)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

class DataSaver:
    @staticmethod
    def save_enhanced_data(original_data, original_labels, enhanced_data, enhanced_labels, feature_names, label_name, save_dir, prefix):
        """
        Saves the generated data, combined dataset, and metadata statistics.
        Outputs: CSV files, NPY file (for raw arrays), and a meta log.
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Generate unique filename based on timestamp
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{prefix}_{timestamp}"
        
        # --- 1. Save Enhanced Data Only ---
        enhanced_df = pd.DataFrame(enhanced_data, columns=feature_names)
        enhanced_df[label_name] = enhanced_labels
        enhanced_df.to_csv(os.path.join(save_dir, f"{base_name}_enhanced.csv"), index=False)
        
        # Save raw numpy array for fast reloading
        np.save(os.path.join(save_dir, f"{base_name}_enhanced.npy"), enhanced_data)
        
        # --- 2. Save Combined Data (Original + Enhanced) ---
        combined_data = np.vstack([original_data, enhanced_data])
        combined_labels = np.concatenate([original_labels, enhanced_labels])
        
        combined_df = pd.DataFrame(combined_data, columns=feature_names)
        combined_df[label_name] = combined_labels
        combined_df.to_csv(os.path.join(save_dir, f"{base_name}_full.csv"), index=False)
        
        # --- 3. Save Metadata & Statistics ---
        meta = {
            'original_samples': int(original_data.shape[0]),
            'enhanced_samples': int(enhanced_data.shape[0]),
            # Calculate mean Wasserstein distance across all feature dimensions
            'w_distance': float(np.mean([wasserstein_distance(original_data[:, i], enhanced_data[:, i]) for i in range(original_data.shape[1])])),
            'generation_time': timestamp
        }
        # Save metadata as a simple CSV series
        pd.Series(meta).to_csv(os.path.join(save_dir, f"{base_name}_meta.csv"))



# ===================== Main Execution Flow =====================
def main():
    # Ensure the save directory exists
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # 1. Load and prepare data
    X, y, feature_names = load_data()
    
    # 2. Initialize Environment and Agent
    env = RLEnvironment(X, y)
    agent = AugmentAgent(X.shape[1])

    # History dictionary to store training metrics for plotting
    history = {
        'episode': [], 'reward': [], 'w_dist': [],
        'alpha_mean': [], 'alpha_std': [], 'beta_mean': [], 'beta_std': []
    }
    
    # Dictionary to track the best model found so far (minimizing Wasserstein distance)
    best = {'w_dist': float('inf'), 'samples': None, 'model': None, 'y': None}
    
    start_time = time.time()
    
    # --- Training Loop ---
    for ep in range(EPISODES):
        # 3. State Preparation
        # Use global state summary (mean of all features) repeated for batch size
        # This assumes a stationary environment or global policy
        states = np.repeat([X.mean(axis=0)], BATCH_SIZE, axis=0)  # Shape: (BATCH_SIZE, dim)
        
        # 4. Action Selection
        # Get raw actions and current value estimates from the Actor-Critic network
        actions_raw, log_probs_torch, values = agent.get_action_and_value(states)
        
        # 5. Monte Carlo Rollouts
        # Estimate returns for each action by simulating multiple futures
        returns = []
        for i in range(len(actions_raw)):
            action = actions_raw[i]
            rollout_returns = []
            
            # Perform MC_ROLLOUTS simulations for variance reduction
            for r in range(MC_ROLLOUTS):
                # Generate a temporary batch based on the current action
                X_new_mc, y_new_mc = DataGenerator.generate_batch(
                    env.X_train, env.y_train, action.reshape(1, -1),
                    feature_constraints=DATA_CONSTRAINTS,
                    label_constraints=LABEL_CONSTRAINTS
                )
                
                # Simulate enhancement WITHOUT modifying the main environment state
                ret = env.simulate_enhancement(X_new_mc, y_new_mc)
                rollout_returns.append(ret)
                
            # Average return from all rollouts acts as the target for the Critic
            avg_ret = float(np.mean(rollout_returns))
            returns.append(avg_ret)
            
        returns = np.array(returns, dtype=float)  # Shape: (batch,)

        # 6. Update Actor-Critic Network
        # Uses the estimated returns to update policy (Actor) and value function (Critic)
        agent.update(states, actions_raw, returns, log_probs_torch, values)

        # 7. Real Evaluation & Environment Step
        # Generate a new batch using the updated policy
        X_new, y_new = DataGenerator.generate_batch(
            env.X_train, env.y_train, actions_raw,
            feature_constraints=DATA_CONSTRAINTS,
            label_constraints=LABEL_CONSTRAINTS
        )
        
        # Evaluate and APPLY the enhancement to the main environment model
        # This updates `env.model` with the new data
        reward = env.evaluate_enhancement(X_new, y_new)

        # 8. Calculate Distribution Distance (Wasserstein) for logging
        try:
            w_dist = np.mean([wasserstein_distance(env.X_train[:, i], X_new[:, i]) for i in range(X.shape[1])])
        except Exception:
            w_dist = float('inf')

        # 9. Log Statistics
        # Map raw actions back to interpretable ranges (alpha/beta) for monitoring
        alphas = 1.0 / (1.0 + np.exp(-actions_raw[:, 0]))
        betas = 0.1 * (np.tanh(actions_raw[:, 1]) + 1.0)
        
        history['episode'].append(ep + 1)
        history['reward'].append(float(reward))
        history['w_dist'].append(float(w_dist))
        history['alpha_mean'].append(float(np.mean(alphas)))
        history['alpha_std'].append(float(np.std(alphas)))
        history['beta_mean'].append(float(np.mean(betas)))
        history['beta_std'].append(float(np.std(betas)))

        # 10. Checkpoint Best Model
        # Save the model that achieves the lowest distribution shift (lowest W_dist)
        if w_dist < best['w_dist']:
            best['w_dist'] = w_dist
            best['samples'] = X_new.copy()
            best['model'] = agent.net.state_dict()
            best['y'] = y_new.copy()

        # Print progress every 10 episodes
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"Episode {ep+1}/{EPISODES}, Reward: {reward:.6f}, W-Dist: {w_dist:.6f}, alpha_mean: {np.mean(alphas):.4f}, beta_mean: {np.mean(betas):.4f}")

    # ===================== Saving Results =====================
    
    # Save training history to CSV
    pd.DataFrame(history).to_csv(os.path.join(SAVE_DIR, f'{RESULT_PREFIX}_history.csv'), index=False)
    
    # Save the best model weights (PyTorch state dict)
    torch.save(best['model'], os.path.join(SAVE_DIR, f'{RESULT_PREFIX}_model.pth'))
    
    if best['samples'] is not None:
        # Save the best generated samples
        np.savez(os.path.join(SAVE_DIR, f'{RESULT_PREFIX}_samples.npz'), X=best['samples'], y=best['y'])
        
        # Visualize results
        ResultVisualizer.plot_pca_comparison(env.X_orig, best['samples'], os.path.join(SAVE_DIR, f'{RESULT_PREFIX}_pca.png'))
        ResultVisualizer.save_training_history(history, os.path.join(SAVE_DIR, f'{RESULT_PREFIX}_training.png'))
        
        # Save full dataset and metadata using the DataSaver utility
        DataSaver.save_enhanced_data(
            original_data=env.X_orig, original_labels=env.y_orig,
            enhanced_data=best['samples'], enhanced_labels=best['y'],
            feature_names=feature_names, label_name=LABEL_COL,
            save_dir=SAVE_DIR, prefix=RESULT_PREFIX
        )
        
    # Calculate and print total elapsed time
    end_time = time.time()
    elapsed_seconds = end_time - start_time
    hrs = int(elapsed_seconds // 3600)
    mins = int((elapsed_seconds % 3600) // 60)
    secs = int(elapsed_seconds % 60)
    elapsed_hms = f"{hrs:02d}:{mins:02d}:{secs:02d}"
    
    print(f"Total Time: {elapsed_seconds:.2f} seconds ({elapsed_hms})")
    print("Training finished. Best W_dist:", best['w_dist'])

if __name__ == "__main__":
    main()