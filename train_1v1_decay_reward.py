"""
Optimized Defender-Attacker Differential Game Implementation
A two-player reinforcement learning setup where both defender and attacker
learn optimal strategies in a competitive zero-sum game.
"""
import numpy as np
import math
from math import sin, cos, acos, asin, sqrt, pi, atan2
import gym
from gym import spaces
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Circle
import time
import os
import json
from datetime import datetime
from scipy.interpolate import RegularGridInterpolator
from scipy.io import loadmat
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm, trange
import torch.multiprocessing as mp
from numba import jit
import cProfile

# Create output directories if they don't exist
os.makedirs('models', exist_ok=True)
os.makedirs('logs', exist_ok=True)
os.makedirs('plots', exist_ok=True)

# Use CUDA if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
print(f"Number of GPUs available: {num_gpus}")

# Set random seeds for reproducibility
RANDOM_SEED = 50
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)

#####################################################################
# UTILITY FUNCTIONS
#####################################################################

def cartesian_oval(xF1, xF2, gamma, rho):
    """
    Computes points along a Cartesian oval given two foci xF1 and xF2.
    Optimized with vectorized operations for faster computation.
    
    Parameters:
      xF1 : array-like, shape (2,)
      xF2 : array-like, shape (2,)
      gamma : float (0 < gamma < 1)
      rho   : float
    
    Returns:
      xsave : ndarray, shape (n,2)  -- each column corresponds to a branch
      ysave : ndarray, shape (n,2)
      xtan1 : ndarray, shape (2,)
      xtan2 : ndarray, shape (2,)
      ang   : float (radians)
    """
    xF1 = np.array(xF1, dtype=np.float32)
    xF2 = np.array(xF2, dtype=np.float32)
    lam = math.atan2(xF1[1] - xF2[1], xF1[0] - xF2[0])
    d = np.linalg.norm(xF2 - xF1)
    
    # Pre-compute constants
    gamma_squared = gamma**2
    p = (1 - gamma_squared) * (d**2 - rho**2)
    p = max(0.0, p)
    
    arg = (np.sqrt(p) - gamma * rho) / d
    arg = np.clip(arg, -1.0, 1.0)
    ang = math.acos(arg)
    
    n = 101
    phi = np.linspace(-ang, ang, n)
    
    # Pre-allocate output arrays
    xsave = np.zeros((n, 2), dtype=np.float32)
    ysave = np.zeros((n, 2), dtype=np.float32)
    
    # Vectorize all calculations
    cos_phi = np.cos(phi)
    q = gamma * rho + d * cos_phi
    diff = q**2 - p
    diff = np.maximum(diff, 0.0)  # Ensure non-negative values
    sqrt_term = np.sqrt(diff)
    
    # Calculate r values for both branches at once
    r_vals_1 = (q + sqrt_term) / (1 - gamma_squared)
    r_vals_2 = (q - sqrt_term) / (1 - gamma_squared)
    
    # Calculate angles
    angle = lam + phi
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    
    # Calculate x and y coordinates for both branches
    # First branch
    xsave[:, 0] = xF2[0] + r_vals_1 * cos_angle
    ysave[:, 0] = xF2[1] + r_vals_1 * sin_angle
    
    # Second branch
    xsave[:, 1] = xF2[0] + r_vals_2 * cos_angle
    ysave[:, 1] = xF2[1] + r_vals_2 * sin_angle
    
    # Set tangent points
    xtan1 = np.array([xsave[0, 0], ysave[0, 0]], dtype=np.float32)
    xtan2 = np.array([xsave[-1, 1], ysave[-1, 1]], dtype=np.float32)
    
    return xsave, ysave, xtan1, xtan2, ang

# JIT-compile the segment intersection check for faster performance
@jit(nopython=True)
def _segment_intersection_fast(x1, y1, x2, y2, x3, y3, x4, y4):
    """
    Check if line segments (x1,y1)-(x2,y2) and (x3,y3)-(x4,y4) intersect.
    Returns 1 if they intersect with the intersection point coordinates, 0 otherwise.
    
    JIT-compiled for significantly faster performance.
    """
    # Convert to line form Ax + By = C
    A1 = y2 - y1
    B1 = x1 - x2
    C1 = A1 * x1 + B1 * y1
    
    A2 = y4 - y3
    B2 = x3 - x4
    C2 = A2 * x3 + B2 * y3
    
    # Calculate determinant
    det = A1 * B2 - A2 * B1
    
    # Check if lines are parallel
    if abs(det) < 1e-8:
        return 0, 0.0, 0.0
    
    # Calculate intersection point
    x = (B2 * C1 - B1 * C2) / det
    y = (A1 * C2 - A2 * C1) / det
    
    # Check if intersection is within both segments
    # Ensure intersection is within first segment
    if min(x1, x2) <= x <= max(x1, x2) and min(y1, y2) <= y <= max(y1, y2):
        # Ensure intersection is within second segment
        if min(x3, x4) <= x <= max(x3, x4) and min(y3, y4) <= y <= max(y3, y4):
            # Exclude endpoints to avoid false positives
            if not ((abs(x - x1) < 1e-8 and abs(y - y1) < 1e-8) or 
                   (abs(x - x2) < 1e-8 and abs(y - y2) < 1e-8)):
                return 1, x, y
    
    return 0, 0.0, 0.0

#####################################################################
# ENVIRONMENT
#####################################################################

class DefenderAttackerGameEnv(gym.Env):
    """
    Environment for a differential game between an attacker and defender,
    where both agents are learning simultaneously.
    """
    def __init__(self, 
                 gamma=0.5,
                 rho=1.0,
                 dt=0.1,
                 render_debug=False,
                 normalize_rewards=True,
                 visualization_interval=500,
                 env_id=0):  # Add env_id for parallel environments
        super(DefenderAttackerGameEnv, self).__init__()
        self.gamma = gamma
        self.rho_param = rho
        self.dt = dt
        self.r_capture = 1.0
        self.target = np.array([0.0, 0.0], dtype=np.float32)
        self.render_debug = render_debug
        self.normalize_rewards = normalize_rewards
        self.visualization_interval = visualization_interval
        self.should_visualize = False
        self.env_id = env_id
        
        # For reward decay over time as agents learn
        self.reward_decay_factor = 1.0  # Default to no decay
        
        # Caching for geometric calculations
        self._last_positions = None
        self._last_blocked_result = None
        self._last_oval = None
        
        # For optimized rendering
        self.fig = None
        self.ax = None
        
        # State: [xA, yA, xD, yD, heading_A, heading_D]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        
        # Action spaces for both agents (heading command in [-pi, pi])
        self.attacker_action_space = spaces.Box(low=-pi, high=pi, shape=(1,), dtype=np.float32)
        self.defender_action_space = spaces.Box(low=-pi, high=pi, shape=(1,), dtype=np.float32)
        
        # For tracking trajectories when visualizing batches
        self.trajectory_buffer = {
            'attacker_x': [],
            'attacker_y': [],
            'defender_x': [],
            'defender_y': []
        }
        
        # Load lookup table if available (for analytical solutions)
        lookup_file = 'lookup_table_value.mat'
        try:
            # Try different possible paths for the lookup table
            possible_paths = [
                lookup_file,  # Current directory
                os.path.join(os.path.dirname(__file__), lookup_file),  # Same directory as script
                os.path.join(os.path.abspath('.'), lookup_file),  # Absolute path to current directory
                os.path.join('..', lookup_file)  # Parent directory
            ]
            
            # Try each path until we find the file
            data = None
            loaded_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    data = loadmat(path)
                    loaded_path = path
                    break
            
            if data is not None:
                self.lookup_RHO = data['RHO']
                self.lookup_THETA = data['THETA']
                self.lookup_V = -data['V']
                self.interpolator = RegularGridInterpolator(
                    (self.lookup_THETA[:, 0], self.lookup_RHO[0, :]), self.lookup_V,
                    bounds_error=False, fill_value=None)
                print(f"Env {self.env_id}: Lookup table loaded successfully from {loaded_path}")
            else:
                print(f"Env {self.env_id}: Lookup table file not found in any expected location")
                self.interpolator = None
        except Exception as e:
            print(f"Env {self.env_id}: Lookup table not loaded, using fallback. Exception: {e}")
            self.interpolator = None
        
        # For tracking progress
        self.episode_step = 0
        self.attacker_total_reward = 0
        self.defender_total_reward = 0
        self.terminal_info = None
        self.intersections = []
        self.defender_target_point = None
        self.reset()
    
    def set_visualization(self, should_visualize):
        """Set whether visualization should be enabled for the current episode"""
        self.should_visualize = should_visualize
        if not should_visualize and self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
    
    def _get_state(self):
        """Return the current state vector for both agents"""
        return np.concatenate([
            self.xA, 
            self.xD, 
            np.array([float(self.heading_A), float(self.heading_D)], dtype=np.float32)
        ]).astype(np.float32)
    
    def _get_attacker_observation(self):
        """Return observation from attacker's perspective (same state for now)"""
        return self._get_state()
    
    def _get_defender_observation(self):
        """Return observation from defender's perspective (same state for now)"""
        return self._get_state()
    
    def _check_line_blocked(self):
        """
        Check if the line of sight from attacker to target is blocked by the Cartesian oval.
        Uses polygon-line intersection with caching for performance.
        """
        # Check if we can use the cached result (positions haven't changed significantly)
        position_key = (tuple(np.round(self.xD, 2)), tuple(np.round(self.xA, 2)))
        if hasattr(self, '_last_positions') and self._last_positions == position_key:
            # Use cached result if available
            return self._last_blocked_result
            
        # Calculate the cartesian oval
        self.xsave, self.ysave, self.xtan1, self.xtan2, self.oval_ang = cartesian_oval(
            self.xD, self.xA, self.gamma, self.rho_param)
        
        # Form closed polygon from the two oval branches
        x_branch1 = self.xsave[:, 0]
        y_branch1 = self.ysave[:, 0]
        x_branch2 = self.xsave[:, 1]
        y_branch2 = self.ysave[:, 1]
        
        # Create closed polygon by concatenating branches
        x_poly = np.concatenate([x_branch1, np.flip(x_branch2)])
        y_poly = np.concatenate([y_branch1, np.flip(y_branch2)])
        
        # LOS from attacker to target
        los_x = [self.xA[0], self.target[0]]
        los_y = [self.xA[1], self.target[1]]
        
        # Check for intersections
        intersections = []
        
        # For each polygon segment, check intersection with LOS
        for i in range(len(x_poly)-1):
            # Polygon segment
            poly_x = [x_poly[i], x_poly[i+1]]
            poly_y = [y_poly[i], y_poly[i+1]]
            
            # Check if segments intersect using JIT-compiled function
            has_intersection, x, y = _segment_intersection_fast(
                los_x[0], los_y[0], los_x[1], los_y[1],  # LOS
                poly_x[0], poly_y[0], poly_x[1], poly_y[1]  # Polygon segment
            )
            
            if has_intersection:
                intersections.append((x, y))
        
        # Also check the last segment connecting back to the first point
        poly_x = [x_poly[-1], x_poly[0]]
        poly_y = [y_poly[-1], y_poly[0]]
        
        has_intersection, x, y = _segment_intersection_fast(
            los_x[0], los_y[0], los_x[1], los_y[1],
            poly_x[0], poly_y[0], poly_x[1], poly_y[1]
        )
        
        if has_intersection:
            intersections.append((x, y))
        
        # Store intersection points for visualization
        self.intersections = intersections
        
        # Cache the result
        self._last_positions = position_key
        self._last_blocked_result = len(intersections) > 0
        
        # Return the result
        return self._last_blocked_result
    
    def reset(self):
        """
        Reset the environment with random positions that satisfy our criteria:
        1. Attacker's line of sight to target is blocked
        2. If unblocked, attacker would reach target before defender (t_a < t_d)
        """
        # Hard-code the target at the origin:
        self.target = np.array([0.0, 0.0], dtype=np.float32)
        
        # Generate random positions until we find one that satisfies our criteria
        max_attempts = 100
        for _ in range(max_attempts):
            # Random defender position (within a reasonable range)
            angle_d = np.random.uniform(-np.pi, np.pi)
            distance_d = np.random.uniform(10.0, 12.0)
            self.xD = np.array([
                distance_d * np.cos(angle_d), 
                distance_d * np.sin(angle_d)
            ], dtype=np.float32)
            
            # Random attacker position (behind the defender from target's perspective)
            angle_offset = np.random.uniform(-np.pi/6, np.pi/6)  # Small offset from defender angle
            angle_a = angle_d + angle_offset
            distance_a = np.random.uniform(distance_d + 1.0, distance_d + 3)  # Slightly further than defender
            self.xA = np.array([
                distance_a * np.cos(angle_a), 
                distance_a * np.sin(angle_a)
            ], dtype=np.float32)
            
            # Calculate times for each agent to reach target
            distance_to_target = np.linalg.norm(self.xA - self.target)
            distance_defender_to_target = np.linalg.norm(self.xD - self.target)
            self.t_a = distance_to_target / 1.0  # Attacker speed = 1.0
            self.t_d = distance_defender_to_target / self.gamma  # Defender speed = gamma
            
            # Check if line of sight is blocked
            self.initial_blocked = self._check_line_blocked()
            
            # Check if our criteria are satisfied: blocked but t_a < t_d
            if self.initial_blocked and self.t_a < self.t_d:
                break
        
        # If we couldn't find a valid position after max attempts, use a default
        if not (self.initial_blocked and self.t_a < self.t_d):
            # Default positions that satisfy our criteria
            self.xD = np.array([-10.0, 0.0], dtype=np.float32)
            self.xA = np.array([-12.0, 0.5], dtype=np.float32)
            
            # Recalculate times and blocking
            distance_to_target = np.linalg.norm(self.xA - self.target)
            distance_defender_to_target = np.linalg.norm(self.xD - self.target)
            self.t_a = distance_to_target / 1.0
            self.t_d = distance_defender_to_target / self.gamma
            self.initial_blocked = self._check_line_blocked()
        
        # Set initial headings toward target
        target_dir_A = self.target - self.xA
        self.heading_A = math.atan2(target_dir_A[1], target_dir_A[0])
        
        target_dir_D = self.target - self.xD
        self.heading_D = math.atan2(target_dir_D[1], target_dir_D[0])

        # Clear trajectory buffer
        self.trajectory_buffer = {
            'attacker_x': [],
            'attacker_y': [],
            'defender_x': [],
            'defender_y': []
        }
        
        # Add initial positions to trajectory buffer
        if self.should_visualize:
            self.trajectory_buffer['attacker_x'].append(float(self.xA[0]))
            self.trajectory_buffer['attacker_y'].append(float(self.xA[1]))
            self.trajectory_buffer['defender_x'].append(float(self.xD[0]))
            self.trajectory_buffer['defender_y'].append(float(self.xD[1]))

        # Reset episode info
        self.done = False
        self.episode_step = 0
        self.attacker_total_reward = 0
        self.defender_total_reward = 0
        self.terminal_info = None
        self.intersections = []
        self.defender_target_point = None
        self.prev_distance_to_target = np.linalg.norm(self.xA - self.target)
        self.initial_distance_to_target = self.prev_distance_to_target
        
        # Reset caching
        self._last_positions = None
        self._last_blocked_result = None

        # Calculate times, check oval, etc.
        self.target_in_oval = self.t_d <= self.t_a

        observations = {
            'attacker': self._get_attacker_observation(),
            'defender': self._get_defender_observation()
        }
        return observations
    
    def step(self, attacker_action, defender_action):
        """Take a step in the environment with actions for both agents"""
        self.episode_step += 1
        
        # Update attacker and defender headings from actions
        self.heading_A = float(attacker_action[0])
        self.heading_D = float(defender_action[0])
        
        # Calculate time to reach target for both agents
        distance_to_target = np.linalg.norm(self.xA - self.target)
        distance_defender_to_target = np.linalg.norm(self.xD - self.target)
        
        t_a = distance_to_target / 1.0  # Attacker's time (speed = 1.0)
        t_d = distance_defender_to_target / self.gamma  # Defender's time
        
        # Determine if target is inside oval based on time relation
        self.target_in_oval = t_d <= t_a
        
        # Check if line of sight is blocked
        at_line_blocked = self._check_line_blocked()
        
        # Update positions
        vA = 1.0  # Attacker speed
        self.xA = self.xA + vA * self.dt * np.array([cos(self.heading_A), sin(self.heading_A)], dtype=np.float32)
        vD = self.gamma  # Defender speed
        self.xD = self.xD + vD * self.dt * np.array([cos(self.heading_D), sin(self.heading_D)], dtype=np.float32)
        
        # Store trajectory data if visualization is enabled
        if self.should_visualize:
            self.trajectory_buffer['attacker_x'].append(float(self.xA[0]))
            self.trajectory_buffer['attacker_y'].append(float(self.xA[1]))
            self.trajectory_buffer['defender_x'].append(float(self.xD[0]))
            self.trajectory_buffer['defender_y'].append(float(self.xD[1]))
        
        # Clear position cache since positions have changed
        self._last_positions = None
        
        # Recalculate metrics after movement
        distance_to_target = np.linalg.norm(self.xA - self.target)
        distance_defender_to_target = np.linalg.norm(self.xD - self.target)
        DA = np.linalg.norm(self.xA - self.xD)  # Distance between attacker and defender
        
        # Determine terminal conditions and reward
        terminal_reward_attacker = 0.0
        terminal_reward_defender = 0.0
        done = False
        
        # Case 1: Defender reaches target
        if distance_defender_to_target <= self.r_capture:
            # Terminal reward for defender is positive (reached goal)
            terminal_reward_defender = distance_to_target * 2.0
            # Terminal reward for attacker is negative 
            terminal_reward_attacker = -distance_to_target * 2.0
            done = True
            self.terminal_info = "defender_reached_target"
            
        # Case 2: Attacker reaches target
        elif distance_to_target < 0.2:
            # Terminal reward for attacker is positive (reached goal)
            terminal_reward_attacker = distance_defender_to_target * 2.0
            # Terminal reward for defender is negative
            terminal_reward_defender = -distance_defender_to_target * 2.0
            done = True
            self.terminal_info = "attacker_reached_target"
        
        # Case 3: Defender captures attacker
        elif DA <= self.r_capture:
            # Base reward structure for capture
            if self.interpolator is not None:
                # Calculate angle between defender-target and defender-attacker vectors
                v1 = self.target - self.xD
                v2 = self.xA - self.xD
                norm1 = np.linalg.norm(v1)
                norm2 = np.linalg.norm(v2)
                dot_product = np.dot(v1, v2)
                theta = acos(np.clip(dot_product / (norm1 * norm2 + 1e-8), -1.0, 1.0))
                
                # Lookup terminal value based on angle and distance
                pt = np.array([theta, distance_defender_to_target])
                V_val = self.interpolator(pt)
                
                # Opposite rewards for each agent
                terminal_reward_defender = float(V_val.item()) * 3.0
                terminal_reward_attacker = -float(V_val.item()) * 3.0
                
                # Add debugging info if requested
                if self.render_debug and self.should_visualize:
                    print(f"CAPTURE: angle={theta:.2f}, V_val={float(V_val.item()):.2f}, reward_D={terminal_reward_defender:.2f}, reward_A={terminal_reward_attacker:.2f}")
            else:
                # If no lookup table, use time difference with a penalty
                capture_reward = 5.0 - (t_d - t_a)
                terminal_reward_defender = capture_reward
                terminal_reward_attacker = -capture_reward
            
            done = True
            self.terminal_info = "attacker_captured"
            
        # Case 4: Defender has time advantage (tD <= tA)
        elif t_d <= t_a:
            # Terminal reward for defender (time advantage)
            terminal_reward_defender = 3.0 + (t_a - t_d)
            # Terminal reward for attacker is negative
            terminal_reward_attacker = -3.0 - (t_a - t_d)
            done = True
            self.terminal_info = "defender_time_advantage"
        
        # Case 5: Unblocked path with time advantage for attacker
        elif not at_line_blocked and not self.target_in_oval:
            # Recalculate times
            t_a = distance_to_target / vA
            t_d = distance_defender_to_target / vD
            
            if t_a < t_d:
                # Terminal reward is defender's distance to target (same as reaching target)
                terminal_reward_attacker = distance_defender_to_target * 2.0
                terminal_reward_defender = -distance_defender_to_target * 2.0
                done = True
                self.terminal_info = "unblocked_path_advantage"
        
        # Case 6: Episode timeout
        elif self.episode_step >= 300:
            # Mild negative for both agents for timeout
            terminal_reward_attacker = -2.0
            terminal_reward_defender = -2.0
            done = True
            self.terminal_info = "timeout"
        
        # Calculate shaped rewards during episode
        if not done:
            # Attacker's shaped reward (progress toward target)
            attacker_reward = self._calculate_attacker_shaped_reward(distance_to_target, DA, at_line_blocked)
            
            # Defender's shaped reward (opposite goals in zero-sum game)
            defender_reward = self._calculate_defender_shaped_reward(distance_to_target, DA, at_line_blocked)
            
            # Apply reward decay factor (will gradually reduce rewards as training progresses)
            attacker_reward *= self.reward_decay_factor
            defender_reward *= self.reward_decay_factor
        else:
            # Only use terminal rewards if episode is done
            attacker_reward = terminal_reward_attacker
            defender_reward = terminal_reward_defender
        
        # Update stored values
        self.prev_distance_to_target = distance_to_target
        self.attacker_total_reward += attacker_reward
        self.defender_total_reward += defender_reward
        self.done = done
        
        # Create info dictionary
        info = {
            "distance_to_target": distance_to_target,
            "defender_to_target": distance_defender_to_target,
            "DA": DA,
            "at_line_blocked": at_line_blocked,
            "target_in_oval": self.target_in_oval,
            "time_attacker": t_a,
            "time_defender": t_d,
            "episode_step": self.episode_step,
            "attacker_total_reward": self.attacker_total_reward,
            "defender_total_reward": self.defender_total_reward,
            "initial_blocked": self.initial_blocked,
            "terminal_condition": self.terminal_info if done else None,
            "terminal_reward_attacker": terminal_reward_attacker if done else 0.0,
            "terminal_reward_defender": terminal_reward_defender if done else 0.0,
        }
        
        observations = {
            'attacker': self._get_attacker_observation(),
            'defender': self._get_defender_observation()
        }
        
        rewards = {
            'attacker': attacker_reward,
            'defender': defender_reward
        }
        
        return observations, rewards, done, info
    
    def _calculate_attacker_shaped_reward(self, distance_to_target, DA, at_line_blocked):
        """
        Streamlined reward for the attacker/evader with priorities:
        1. Get closer to the target (primary)
        2. Increase theta to avoid blocking (secondary)
        3. Minimize path length and time to target (tertiary)
        """
        # Normalization for rewards
        if self.normalize_rewards:
            normalization_factor = max(self.initial_distance_to_target, 1e-8)
        else:
            normalization_factor = 1.0
        
        # Scale factor
        scale_factor = 0.05
        reward = 0.0
        
        # 1. Primary goal: Progress toward target
        progress_weight = 10.0
        dist_progress = (self.prev_distance_to_target - distance_to_target) / normalization_factor
        progress_reward = scale_factor * progress_weight * dist_progress
        reward += progress_reward
        
        # 2. Secondary goal: Increase theta to avoid blocking
        theta_weight = 5.0
        
        # Calculate theta (angle between defender-target and defender-attacker vectors)
        v1 = self.target - self.xD  # Vector from defender to target
        v2 = self.xA - self.xD      # Vector from defender to attacker
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        dot_product = np.dot(v1, v2)
        theta = math.acos(np.clip(dot_product / (norm1 * norm2 + 1e-8), -1.0, 1.0))
        
        # Only reward theta increase if path is blocked
        if at_line_blocked:
            # Optimal theta is around pi/2 (90 degrees)
            optimal_theta = np.pi/2
            theta_diff = abs(theta - optimal_theta)
            theta_reward = scale_factor * theta_weight * (1.0 - theta_diff / optimal_theta)
            reward += theta_reward
        
        # 3. Tertiary goal: Minimize path length / time to target
        efficiency_weight = 3.0
        
        # Reward for heading directly toward target
        target_vector = self.target - self.xA
        target_heading = math.atan2(target_vector[1], target_vector[0])
        heading_alignment = abs(target_heading - self.heading_A)
        heading_alignment = min(heading_alignment, 2*np.pi - heading_alignment)
        
        if not at_line_blocked:
            # If path is clear, reward direct movement toward target
            efficiency_reward = scale_factor * efficiency_weight * (1.0 - heading_alignment / np.pi)
            reward += efficiency_reward
        
        # Small time penalty to encourage speed
        reward -= scale_factor / 20
        
        return reward
    
    def _calculate_defender_shaped_reward(self, distance_to_target, DA, at_line_blocked):
        """
        Streamlined reward for the defender/pursuer with priorities:
        1. Block attacker's line of sight to target 
        2. Maximize attacker's path length and time
        3. Capture attacker with smaller theta
        4. Move toward target with balanced priority
        """
        if self.normalize_rewards:
            normalization_factor = max(self.initial_distance_to_target, 1e-8)
        else:
            normalization_factor = 1.0
        
        # Scale factor
        scale_factor = 0.05
        reward = 0.0
        
        # 1. Primary goal: Block line of sight
        blocking_weight = 10.0
        
        # If currently blocking, reward staying in blocking position
        if at_line_blocked:
            blocking_reward = scale_factor * blocking_weight
            reward += blocking_reward
        else:
            # Calculate direction to move to block
            attacker_to_target = self.target - self.xA
            attacker_to_target_dir = attacker_to_target / np.linalg.norm(attacker_to_target)
            
            # Calculate projection of defender position onto attacker-target line
            attacker_to_defender = self.xD - self.xA
            projection = np.dot(attacker_to_defender, attacker_to_target_dir)
            
            # Calculate the closest point on the line
            closest_point = self.xA + projection * attacker_to_target_dir
            
            # Calculate heading alignment with direction to blocking position
            direction_to_blocking = closest_point - self.xD
            direction_to_blocking_heading = math.atan2(direction_to_blocking[1], direction_to_blocking[0])
            heading_alignment = abs(direction_to_blocking_heading - self.heading_D)
            heading_alignment = min(heading_alignment, 2*np.pi - heading_alignment)
            
            # Reward for alignment with blocking position
            alignment_reward = scale_factor * blocking_weight * 0.5 * (1.0 - heading_alignment / np.pi)
            reward += alignment_reward
        
        # 2. Secondary goal: Maximize attacker's path length and time
        path_weight = 5.0
        
        # Calculate theta (angle between defender-target and defender-attacker vectors)
        v1 = self.target - self.xD  # Vector from defender to target
        v2 = self.xA - self.xD      # Vector from defender to attacker
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        dot_product = np.dot(v1, v2)
        theta = math.acos(np.clip(dot_product / (norm1 * norm2 + 1e-8), -1.0, 1.0))
        
        # For maximizing path length, the defender wants to be between attacker and target
        # with smaller theta (forcing attacker to take a longer path)
        if not at_line_blocked:
            # Reward smaller theta (opposite of attacker's goal)
            path_reward = scale_factor * path_weight * (1.0 - theta / (np.pi/2))
            reward += path_reward
        
        # 3. Tertiary goal: Capture attacker with smaller theta
        capture_weight = 4.0
        
        # Only apply when close enough to consider capture
        if DA < self.r_capture * 3.0:
            # For capture, reward smaller theta
            capture_reward = scale_factor * capture_weight * (1.0 - theta / (np.pi/2)) * (1.0 - DA / (self.r_capture * 3.0))
            reward += capture_reward
        
        # 4. Quaternary goal: Move toward target with balance
        target_weight = 2.0
        
        # Calculate heading alignment with direction to target
        direction_to_target = self.target - self.xD
        direction_to_target_heading = math.atan2(direction_to_target[1], direction_to_target[0])
        target_heading_alignment = abs(direction_to_target_heading - self.heading_D)
        target_heading_alignment = min(target_heading_alignment, 2*np.pi - target_heading_alignment)
        
        # Only reward moving toward target if other goals are being met
        if at_line_blocked or DA < self.r_capture * 2.0:
            target_reward = scale_factor * target_weight * (1.0 - target_heading_alignment / np.pi)
            reward += target_reward
        
        # Small time penalty to encourage action
        reward -= scale_factor / 20
        
        return reward
    
    def render(self, mode='human'):
        """Renders the current state of the environment with reused figure"""
        if not self.render_debug or not self.should_visualize:
            return
            
        # Create figure only once, reuse it for performance
        if self.fig is None or not plt.fignum_exists(self.fig.number):
            self.fig, self.ax = plt.subplots(figsize=(10, 10))
        else:
            self.ax.clear()
            
        # Calculate time-to-target for both agents
        distance_to_target = np.linalg.norm(self.xA - self.target)
        distance_defender_to_target = np.linalg.norm(self.xD - self.target)
        t_a = distance_to_target / 1.0  # Attacker's time
        t_d = distance_defender_to_target / self.gamma  # Defender's time
            
        # Plot Cartesian oval (use cached calculation if available)
        if hasattr(self, 'xsave') and hasattr(self, 'ysave'):
            # Use cached oval from _check_line_blocked
            xsave, ysave = self.xsave, self.ysave
        else:
            # Calculate it if not cached
            xsave, ysave, _, _, _ = cartesian_oval(self.xD, self.xA, self.gamma, self.rho_param)
        
        # Create a closed curve representing the oval
        x_branch1 = xsave[:, 0]
        y_branch1 = ysave[:, 0]
        x_branch2 = xsave[:, 1]
        y_branch2 = ysave[:, 1]
        
        self.ax.plot(x_branch1, y_branch1, 'r-', alpha=0.6, linewidth=1, label='Oval Branch 1')
        self.ax.plot(x_branch2, y_branch2, 'r-', alpha=0.6, linewidth=1, label='Oval Branch 2')
        
        # Compute line of sight from attacker to target
        self.ax.plot([self.xA[0], self.target[0]], [self.xA[1], self.target[1]], 'g--', alpha=0.6, label='LOS')
        
        # Plot target
        self.ax.plot(self.target[0], self.target[1], 'g*', markersize=15, label='Target')
        
        # Plot attacker
        self.ax.plot(self.xA[0], self.xA[1], 'ro', markersize=10, label='Attacker')
        
        # Plot defender and capture radius
        self.ax.plot(self.xD[0], self.xD[1], 'bo', markersize=10, label='Defender')
        defender_circle = plt.Circle((self.xD[0], self.xD[1]), self.r_capture, color='b', fill=False, alpha=0.3)
        self.ax.add_patch(defender_circle)
        
        # Plot trajectories if available
        if len(self.trajectory_buffer['attacker_x']) > 1:
            self.ax.plot(
                self.trajectory_buffer['attacker_x'], 
                self.trajectory_buffer['attacker_y'], 
                'r-', alpha=0.5, linewidth=1.5, label='Attacker Path'
            )
            self.ax.plot(
                self.trajectory_buffer['defender_x'], 
                self.trajectory_buffer['defender_y'], 
                'b-', alpha=0.5, linewidth=1.5, label='Defender Path'
            )
        
        # Safety radius visualization (1.5 * capture radius)
        safety_circle = plt.Circle((self.xD[0], self.xD[1]), self.r_capture * 1.5, color='b', fill=False, alpha=0.1, linestyle='--')
        self.ax.add_patch(safety_circle)
        
        # Show heading vectors
        self.ax.arrow(self.xA[0], self.xA[1], 
                cos(self.heading_A)*0.8, sin(self.heading_A)*0.8, 
                color='r', width=0.05, head_width=0.3, alpha=0.7)
        self.ax.arrow(self.xD[0], self.xD[1], 
                cos(self.heading_D)*0.8, sin(self.heading_D)*0.8, 
                color='b', width=0.05, head_width=0.3, alpha=0.7)
        
        # Plot any detected intersections
        if hasattr(self, 'intersections') and self.intersections:
            for idx, point in enumerate(self.intersections):
                self.ax.plot(point[0], point[1], 'rx', markersize=10, 
                           label=f'Intersection {idx+1}' if idx == 0 else None)
        
        # Show episode information with time comparison
        blocked_status = "BLOCKED" if self._check_line_blocked() else "UNBLOCKED"
        time_advantage = "DEFENDER FASTER" if t_d <= t_a else "ATTACKER FASTER"
        terminal_info = f", Outcome: {self.terminal_info}" if self.done else ""
        self.ax.set_title(f'Step: {self.episode_step}, A Reward: {self.attacker_total_reward:.2f}, D Reward: {self.defender_total_reward:.2f}{terminal_info}\nPath: {blocked_status}, {time_advantage} (tA={t_a:.1f}, tD={t_d:.1f})')
        
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        
        # Adjust limits to provide better view
        all_x = [self.xA[0], self.xD[0], self.target[0]]
        all_y = [self.xA[1], self.xD[1], self.target[1]]
        
        # Include trajectory points if available
        if len(self.trajectory_buffer['attacker_x']) > 0:
            all_x.extend(self.trajectory_buffer['attacker_x'])
            all_y.extend(self.trajectory_buffer['attacker_y'])
            all_x.extend(self.trajectory_buffer['defender_x'])
            all_y.extend(self.trajectory_buffer['defender_y'])
        
        max_range = max(np.ptp(all_x), np.ptp(all_y)) * 1.2
        mid_x = np.mean(all_x)
        mid_y = np.mean(all_y)
        self.ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
        self.ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
        
        self.ax.legend(loc='upper right')
        
        # Update canvas for efficiency
        self.fig.canvas.draw()
        plt.pause(0.01)

#####################################################################
# REPLAY BUFFER
#####################################################################

class ReplayBuffer:
    def __init__(self, capacity, state_dim=None, action_dim=None, seed=None):
        self.capacity = capacity
        
        # Always use pre-allocated arrays for better performance
        assert state_dim is not None and action_dim is not None, "Must provide state_dim and action_dim"
        # Pre-allocated implementation
        self.use_prealloc = True
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.size = 0
        
        self.pos = 0
        self.rng = np.random.RandomState(seed)
        
    def push(self, state, action, reward, next_state, done):
        """Store a transition in the buffer (optimized with pre-allocated arrays)"""
        # Store directly in the pre-allocated arrays
        self.states[self.pos] = state
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_states[self.pos] = next_state
        self.dones[self.pos] = float(done)
        
        # Update position and size
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size):
        """Sample a batch of transitions (optimized for efficiency)"""
        # Optimized sampling from pre-allocated arrays
        indices = self.rng.choice(self.size, batch_size, replace=False)
        return (
            torch.FloatTensor(self.states[indices]).to(device),
            torch.FloatTensor(self.actions[indices]).to(device),
            torch.FloatTensor(self.rewards[indices]).to(device),
            torch.FloatTensor(self.next_states[indices]).to(device),
            torch.FloatTensor(self.dones[indices]).to(device)
        )
    
    def __len__(self):
        """Return the current size of the buffer"""
        return self.size

#####################################################################
# SAC AGENT IMPLEMENTATION
#####################################################################

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        """
        Actor network for SAC with improved architecture
        """
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LeakyReLU(0.1),  # LeakyReLU for better gradient flow
            nn.LayerNorm(hidden_dim),  # LayerNorm for training stability
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(hidden_dim)
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
        # Initialize weights for better training
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        """Initialize network weights for better training stability"""
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight.data, gain=np.sqrt(2))
            nn.init.constant_(m.bias.data, 0.0)
    
    def forward(self, state):
        """Forward pass through the network"""
        x = self.net(state)
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2)  # Prevent numerical issues
        return mean, log_std
    
    def sample(self, state):
        """Sample actions using the reparameterization trick"""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        # Sample using reparameterization trick
        epsilon = torch.randn_like(mean).to(device)
        x_t = mean + epsilon * std
        
        y_t = torch.tanh(x_t)
        # Scale from [-1, 1] to [-pi, pi]
        action = y_t * pi
        
        # Optimized log_prob calculation
        log_prob = -0.5 * ((epsilon ** 2) + 2 * log_std + math.log(2 * math.pi))
        log_prob = log_prob - torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return action, log_prob

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        """
        Q-Network for SAC with improved architecture
        """
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        """Initialize network weights"""
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight.data, gain=np.sqrt(2))
            nn.init.constant_(m.bias.data, 0.0)
            
    def forward(self, state, action):
        """Forward pass through the Q-network"""
        x = torch.cat([state, action], dim=1)
        return self.net(x)

class SACAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=256, 
                 actor_lr=3e-4, critic_lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2):
        """
        Soft Actor-Critic agent with optimized implementations
        """
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        # Initialize networks
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(device)
        self.q1 = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.q2 = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.q1_target = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.q2_target = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        
        # Use DataParallel if multiple GPUs are available
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for actor/critic networks")
            self.actor = nn.DataParallel(self.actor)
            self.q1 = nn.DataParallel(self.q1)
            self.q2 = nn.DataParallel(self.q2)
            self.q1_target = nn.DataParallel(self.q1_target)
            self.q2_target = nn.DataParallel(self.q2_target)
        
        # Initialize target networks with source network parameters
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # Initialize optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=critic_lr)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=critic_lr)
        
        # For tracking losses
        self.actor_losses = []
        self.critic1_losses = []
        self.critic2_losses = []
        self.q_values = []
        
    def select_action(self, state, evaluate=False):
        """
        Select an action given the current state
        Args:
            state: The current state
            evaluate: If True, use the mean action instead of sampling
        Returns:
            action: The selected action
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        with torch.no_grad():
            if evaluate:
                # During evaluation, use the mean action
                mean, _ = self.actor(state)
                action = torch.tanh(mean) * pi
            else:
                # During training, sample from the policy
                action, _ = self.actor.sample(state)
                
        return action.cpu().numpy().flatten()
        
    def update(self, batch):
        """
        Update the agent's networks using the given batch of experiences
        Args:
            batch: Tuple of (states, actions, rewards, next_states, dones)
        """
        states, actions, rewards, next_states, dones = batch
        
        # Update Q-networks
        with torch.no_grad():
            # Sample actions from the target policy
            next_actions, next_log_probs = self.actor.sample(next_states)
            
            # Compute target Q-values
            next_q1 = self.q1_target(next_states, next_actions)
            next_q2 = self.q2_target(next_states, next_actions)
            next_q = torch.min(next_q1, next_q2) - self.alpha * next_log_probs
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Compute current Q-values
        current_q1 = self.q1(states, actions)
        current_q2 = self.q2(states, actions)
        
        # Compute critic losses
        q1_loss = F.mse_loss(current_q1, target_q)
        q2_loss = F.mse_loss(current_q2, target_q)
        
        # Update critics
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()
        
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()
        
        # Update actor
        actions_pred, log_probs = self.actor.sample(states)
        q1_pred = self.q1(states, actions_pred)
        q2_pred = self.q2(states, actions_pred)
        q_pred = torch.min(q1_pred, q2_pred)
        
        actor_loss = (self.alpha * log_probs - q_pred).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update target networks
        self.update_target_networks()
        
        # Store metrics for logging
        self.actor_losses.append(actor_loss.item())
        self.critic1_losses.append(q1_loss.item())
        self.critic2_losses.append(q2_loss.item())
        self.q_values.append(current_q1.mean().item())
        
    def update_target_networks(self):
        """Soft update of target networks for improved stability"""
        # Soft update technique - more efficient than full copy
        for target_param, param in zip(self.q1_target.parameters(), self.q1.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )
        
        for target_param, param in zip(self.q2_target.parameters(), self.q2.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )
            
    def save(self, filename):
        """Save the agent's models"""
        torch.save({
            'actor': self.actor.state_dict(),
            'q1': self.q1.state_dict(),
            'q2': self.q2.state_dict(),
            'q1_target': self.q1_target.state_dict(),
            'q2_target': self.q2_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'q1_optimizer': self.q1_optimizer.state_dict(),
            'q2_optimizer': self.q2_optimizer.state_dict(),
        }, filename)
        
    def load(self, filename):
        """Load the agent's models"""
        checkpoint = torch.load(filename)
        self.actor.load_state_dict(checkpoint['actor'])
        self.q1.load_state_dict(checkpoint['q1'])
        self.q2.load_state_dict(checkpoint['q2'])
        self.q1_target.load_state_dict(checkpoint['q1_target'])
        self.q2_target.load_state_dict(checkpoint['q2_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.q1_optimizer.load_state_dict(checkpoint['q1_optimizer'])
        self.q2_optimizer.load_state_dict(checkpoint['q2_optimizer'])

#####################################################################
# PARALLEL ENVIRONMENT WRAPPER
#####################################################################

class ParallelEnvs:
    """A wrapper for parallel environments to speed up training"""
    def __init__(self, env_fn, num_envs=4):
        self.envs = [env_fn(i) for i in range(num_envs)]
        self.num_envs = num_envs
        
    def reset(self):
        """Reset all environments"""
        observations = [env.reset() for env in self.envs]
        attacker_obs = np.array([obs['attacker'] for obs in observations])
        defender_obs = np.array([obs['defender'] for obs in observations])
        
        return {
            'attacker': attacker_obs,
            'defender': defender_obs
        }
    
    def step(self, attacker_actions, defender_actions):
        """Take steps in all environments"""
        results = [env.step(attacker_actions[i], defender_actions[i]) 
                  for i, env in enumerate(self.envs)]
        
        next_observations, rewards, dones, infos = zip(*results)
        
        # Combine results
        next_attacker_obs = np.array([obs['attacker'] for obs in next_observations])
        next_defender_obs = np.array([obs['defender'] for obs in next_observations])
        
        attacker_rewards = np.array([rew['attacker'] for rew in rewards])
        defender_rewards = np.array([rew['defender'] for rew in rewards])
        
        next_observations = {
            'attacker': next_attacker_obs,
            'defender': next_defender_obs
        }
        
        rewards = {
            'attacker': attacker_rewards,
            'defender': defender_rewards
        }
        
        return next_observations, rewards, np.array(dones), infos
    
    def set_visualization(self, should_visualize):
        """Set visualization settings for all environments"""
        for env in self.envs:
            env.set_visualization(should_visualize)

#####################################################################
# DUAL TRAINING SYSTEM
#####################################################################

def train_dual_agents(attacker_agent, defender_agent, env, 
                  attacker_buffer, defender_buffer, 
                  num_episodes, batch_size, updates_per_step, 
                  save_interval=100, eval_interval=100, vis_interval=500,
                  model_dir='models', log_dir='logs',
                  use_parallel=True, num_envs=4):
    """
    Train both attacker and defender agents simultaneously in a competitive setting
    
    Args:
        attacker_agent: The SAC agent for the attacker
        defender_agent: The SAC agent for the defender
        env: The game environment or parallel environment wrapper
        attacker_buffer: Experience replay buffer for attacker
        defender_buffer: Experience replay buffer for defender
        num_episodes: Number of episodes to train for
        batch_size: Batch size for training
        updates_per_step: Number of updates per environment step
        save_interval: How often to save the models
        eval_interval: How often to evaluate the agents
        vis_interval: How often to visualize episodes
        model_dir: Directory to save models
        log_dir: Directory to save logs
        use_parallel: Whether to use parallel environments for data collection
        num_envs: Number of parallel environments to use
    """
    # Start the profiler
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Initialize metrics tracking
    training_metrics = {
        'attacker': {
            'actor_loss': [],
            'critic1_loss': [],
            'critic2_loss': [],
            'avg_q_values': [],
            'episode_rewards': [],
            'success_rate': []
        },
        'defender': {
            'actor_loss': [],
            'critic1_loss': [],
            'critic2_loss': [],
            'avg_q_values': [],
            'episode_rewards': [],
            'success_rate': []
        },
        'global': {
            'attacker_wins': 0,
            'defender_wins': 0,
            'timeouts': 0,
            'terminal_conditions': {
                'attacker_reached_target': 0,
                'defender_reached_target': 0,
                'attacker_captured': 0,
                'defender_time_advantage': 0,
                'unblocked_path_advantage': 0,
                'timeout': 0
            }
        }
    }
    
    # Create timestamp for this training run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    attacker_model_path = os.path.join(model_dir, f"attacker_agent_{timestamp}")
    defender_model_path = os.path.join(model_dir, f"defender_agent_{timestamp}")
    log_path = os.path.join(log_dir, f"dual_training_log_{timestamp}.json")
    
    # Create visualization directory
    vis_dir = os.path.join('plots', timestamp)
    os.makedirs(vis_dir, exist_ok=True)
    
    # Track recent rewards
    attacker_recent_rewards = []
    defender_recent_rewards = []
    
    # Create wrapper for parallel environments if enabled
    if use_parallel and not isinstance(env, ParallelEnvs):
        # Create parallel environment wrapper
        state_dim = env.observation_space.shape[0]
        action_dim = 1
        print(f"Using {num_envs} parallel environments for data collection")
        
        # Create a function to initialize environments
        def make_env(env_id):
            return DefenderAttackerGameEnv(
                gamma=env.gamma,
                rho=env.rho_param,
                dt=env.dt,
                render_debug=env.render_debug,
                normalize_rewards=env.normalize_rewards,
                visualization_interval=vis_interval,
                env_id=env_id
            )
        
        # Create parallel environments
        parallel_env = ParallelEnvs(make_env, num_envs=num_envs)
        env = parallel_env
    
    # Keep track of total steps across all episodes
    total_steps = 0
    completed_episodes = 0
    
    # Create a progress bar for training
    pbar = tqdm(total=num_episodes, desc="Training Progress")
    pbar.update(0)
    
    # Training loop
    while completed_episodes < num_episodes:
        # Determine if this episode should be visualized
        should_visualize = (completed_episodes % vis_interval == 0)
        env.set_visualization(should_visualize)
        
        # Set reward decay factor based on episode count
        # Rewards start high and gradually decay as training progresses
        decay_factor = max(0.2, 1.0 - 0.8 * (completed_episodes / num_episodes))
        if isinstance(env, ParallelEnvs):
            for e in env.envs:
                e.reward_decay_factor = decay_factor
        else:
            env.reward_decay_factor = decay_factor
        
        # Reset environment to get initial observations
        observations = env.reset()
        
        # For parallel environments, we're handling multiple episodes at once
        is_parallel = isinstance(env, ParallelEnvs)
        num_parallel = env.num_envs if is_parallel else 1
        
        attacker_episode_rewards = np.zeros(num_parallel)
        defender_episode_rewards = np.zeros(num_parallel)
        episode_steps = np.zeros(num_parallel, dtype=int)
        dones = np.zeros(num_parallel, dtype=bool)
        
        # Episode loop (handles multiple episodes in parallel if using ParallelEnvs)
        while not np.all(dones):
            # Select actions for all environments
            if is_parallel:
                # Handle parallel environments
                attacker_actions = np.array([
                    attacker_agent.select_action(observations['attacker'][i]) 
                    for i in range(num_parallel) if not dones[i]
                ])
                defender_actions = np.array([
                    defender_agent.select_action(observations['defender'][i])
                    for i in range(num_parallel) if not dones[i]
                ])
                
                # Pad actions for completed environments
                all_attacker_actions = np.zeros((num_parallel, 1))
                all_defender_actions = np.zeros((num_parallel, 1))
                active_idx = 0
                for i in range(num_parallel):
                    if not dones[i]:
                        all_attacker_actions[i] = attacker_actions[active_idx]
                        all_defender_actions[i] = defender_actions[active_idx]
                        active_idx += 1
                
                # Take step in environments
                next_observations, rewards, new_dones, infos = env.step(all_attacker_actions, all_defender_actions)
                
                # Process each environment's results
                for i in range(num_parallel):
                    if not dones[i]:  # Skip already completed environments
                        # Store transitions in respective replay buffers
                        attacker_buffer.push(
                            observations['attacker'][i], 
                            all_attacker_actions[i], 
                            rewards['attacker'][i], 
                            next_observations['attacker'][i], 
                            new_dones[i]
                        )
                        defender_buffer.push(
                            observations['defender'][i], 
                            all_defender_actions[i], 
                            rewards['defender'][i], 
                            next_observations['defender'][i], 
                            new_dones[i]
                        )
                        
                        # Update rewards and steps
                        attacker_episode_rewards[i] += rewards['attacker'][i]
                        defender_episode_rewards[i] += rewards['defender'][i]
                        episode_steps[i] += 1
                        total_steps += 1
                        
                        # Handle episode completion
                        if new_dones[i]:
                            dones[i] = True
                            completed_episodes += 1
                            
                            # Record episode results
                            attacker_recent_rewards.append(attacker_episode_rewards[i])
                            defender_recent_rewards.append(defender_episode_rewards[i])
                            
                            # Record terminal condition
                            condition = infos[i]['terminal_condition']
                            if condition in training_metrics['global']['terminal_conditions']:
                                training_metrics['global']['terminal_conditions'][condition] += 1
                            
                            # Track wins for each agent
                            if condition in ['attacker_reached_target', 'unblocked_path_advantage']:
                                training_metrics['global']['attacker_wins'] += 1
                            elif condition in ['defender_reached_target', 'attacker_captured', 'defender_time_advantage']:
                                training_metrics['global']['defender_wins'] += 1
                            elif condition == 'timeout':
                                training_metrics['global']['timeouts'] += 1
                                
                            # Update progress bar
                            pbar.update(1)
                            pbar.set_description(f"Training - A wins: {training_metrics['global']['attacker_wins']}, D wins: {training_metrics['global']['defender_wins']}, Timeouts: {training_metrics['global']['timeouts']}")
                
                # Update observations
                observations = next_observations
                
            else:
                # Non-parallel (single environment) case
                attacker_action = attacker_agent.select_action(observations['attacker'])
                defender_action = defender_agent.select_action(observations['defender'])
                
                # Take step in environment
                next_observations, rewards, done, info = env.step(attacker_action, defender_action)
                
                # Store transitions in respective replay buffers
                attacker_buffer.push(
                    observations['attacker'], 
                    attacker_action, 
                    rewards['attacker'], 
                    next_observations['attacker'], 
                    done
                )
                defender_buffer.push(
                    observations['defender'], 
                    defender_action, 
                    rewards['defender'], 
                    next_observations['defender'], 
                    done
                )
                
                # Update state and counters
                observations = next_observations
                attacker_episode_rewards[0] += rewards['attacker']
                defender_episode_rewards[0] += rewards['defender']
                episode_steps[0] += 1
                total_steps += 1
                
                # Render if needed (only for visualized episodes)
                if should_visualize:
                    env.render()
                
                # Handle episode completion
                if done:
                    dones[0] = True
                    completed_episodes += 1
                    
                    # Record episode results
                    attacker_recent_rewards.append(attacker_episode_rewards[0])
                    defender_recent_rewards.append(defender_episode_rewards[0])
                    
                    # Record terminal condition
                    condition = info['terminal_condition']
                    if condition in training_metrics['global']['terminal_conditions']:
                        training_metrics['global']['terminal_conditions'][condition] += 1
                    
                    # Track wins for each agent
                    if condition in ['attacker_reached_target', 'unblocked_path_advantage']:
                        training_metrics['global']['attacker_wins'] += 1
                    elif condition in ['defender_reached_target', 'attacker_captured', 'defender_time_advantage']:
                        training_metrics['global']['defender_wins'] += 1
                    elif condition == 'timeout':
                        training_metrics['global']['timeouts'] += 1
                        
                    # Update progress bar
                    pbar.update(1)
                    pbar.set_description(f"Training - A wins: {training_metrics['global']['attacker_wins']}, D wins: {training_metrics['global']['defender_wins']}, Timeouts: {training_metrics['global']['timeouts']}")
            
            # Update networks (with dynamic update frequency based on buffer size)
            if len(attacker_buffer) > batch_size and len(defender_buffer) > batch_size:
                # Dynamically adjust updates_per_step based on replay buffer size
                buffer_size = min(len(attacker_buffer), len(defender_buffer))
                dynamic_updates = max(1, min(4, buffer_size // 10000))
                actual_updates = max(updates_per_step, dynamic_updates)
                
                for _ in range(actual_updates):
                    # Update attacker agent
                    attacker_batch = attacker_buffer.sample(batch_size)
                    attacker_agent.update(attacker_batch)
                    
                    # Update defender agent
                    defender_batch = defender_buffer.sample(batch_size)
                    defender_agent.update(defender_batch)
        
        # Log episode results periodically
        if completed_episodes % 100 == 0:
            avg_attacker_reward = np.mean(attacker_recent_rewards[-10:])
            avg_defender_reward = np.mean(defender_recent_rewards[-10:])
            
            # Calculate win rates
            total_episodes = training_metrics['global']['attacker_wins'] + \
                             training_metrics['global']['defender_wins'] + \
                             training_metrics['global']['timeouts']
                            
            attacker_win_rate = training_metrics['global']['attacker_wins'] / max(total_episodes, 1) * 100
            defender_win_rate = training_metrics['global']['defender_wins'] / max(total_episodes, 1) * 100
            
            print(f"Episode {completed_episodes}/{num_episodes}, Total Steps: {total_steps}")
            print(f"  Attacker Avg Reward: {avg_attacker_reward:.2f}, Win Rate: {attacker_win_rate:.1f}%")
            print(f"  Defender Avg Reward: {avg_defender_reward:.2f}, Win Rate: {defender_win_rate:.1f}%")
            print(f"  Reward Decay Factor: {decay_factor:.2f}")
            
            # Print outcome statistics
            print("Terminal conditions:")
            for condition, count in training_metrics['global']['terminal_conditions'].items():
                print(f"  {condition}: {count} ({count/max(total_episodes, 1)*100:.1f}%)")
            
            # Collect training metrics for both agents
            for agent_type, agent in [('attacker', attacker_agent), ('defender', defender_agent)]:
                if hasattr(agent, 'actor_losses') and len(agent.actor_losses) > 0:
                    training_metrics[agent_type]['actor_loss'].append(np.mean(agent.actor_losses))
                    training_metrics[agent_type]['critic1_loss'].append(np.mean(agent.critic1_losses))
                    training_metrics[agent_type]['critic2_loss'].append(np.mean(agent.critic2_losses))
                    training_metrics[agent_type]['avg_q_values'].append(np.mean(agent.q_values))
                    
                    # Reset metrics lists to avoid memory growth
                    agent.actor_losses = []
                    agent.critic1_losses = []
                    agent.critic2_losses = []
                    agent.q_values = []
            
            # Store average rewards
            training_metrics['attacker']['episode_rewards'].append(avg_attacker_reward)
            training_metrics['defender']['episode_rewards'].append(avg_defender_reward)
            
        # Save models periodically
        if completed_episodes % save_interval == 0:
            attacker_agent.save(f"{attacker_model_path}_episode_{completed_episodes}.pt")
            defender_agent.save(f"{defender_model_path}_episode_{completed_episodes}.pt")
            
            # Save training metrics
            with open(log_path, 'w') as f:
                json.dump(training_metrics, f, indent=2)
        
        # Evaluate agents periodically
        if completed_episodes % eval_interval == 0:
            attacker_success, defender_success = evaluate_dual_agents(
                attacker_agent, defender_agent, env, num_episodes=100)
            
            training_metrics['attacker']['success_rate'].append(attacker_success)
            training_metrics['defender']['success_rate'].append(defender_success)
            
            print(f"Evaluation: Attacker Success = {attacker_success:.1f}%, Defender Success = {defender_success:.1f}%")
    
    # Disable and save profiler results
    profiler.disable()
    profiler.dump_stats(os.path.join(log_dir, f'profile_{timestamp}.prof'))
    
    # Save final models
    attacker_agent.save(f"{attacker_model_path}_final.pt")
    defender_agent.save(f"{defender_model_path}_final.pt")
    
    # Close the progress bar
    pbar.close()
    
    # Save final training metrics
    with open(log_path, 'w') as f:
        json.dump(training_metrics, f, indent=2)
    
    return training_metrics

def evaluate_dual_agents(attacker_agent, defender_agent, env, num_episodes=100, show_progress=True):
    """
    Evaluate both agents against each other
    
    Args:
        attacker_agent: The attacker agent to evaluate
        defender_agent: The defender agent to evaluate
        env: The evaluation environment
        num_episodes: Number of episodes to evaluate
    
    Returns:
        attacker_success_rate: Percentage of episodes where attacker succeeded
        defender_success_rate: Percentage of episodes where defender succeeded
    """
    # Check if we're using parallel environments
    is_parallel = isinstance(env, ParallelEnvs)
    
    # If we have parallel envs, use them to speed up evaluation
    if is_parallel:
        eval_envs = env
        num_envs = eval_envs.num_envs
        num_batches = (num_episodes + num_envs - 1) // num_envs  # Ceiling division
        
        attacker_wins = 0
        defender_wins = 0
        timeouts = 0
        episodes_done = 0
        
        # Create evaluation progress bar if requested
        if show_progress:
            eval_pbar = tqdm(total=num_episodes, desc="Evaluating Agents")
        
        for _ in range(num_batches):
            # Reset environments
            observations = eval_envs.reset()
            
            # Keep track of which environments are done
            dones = np.zeros(num_envs, dtype=bool)
            
            # Run until all environments are done
            while not np.all(dones):
                # Use mean actions without exploration
                attacker_actions = np.array([
                    attacker_agent.select_action(observations['attacker'][i], evaluate=True) 
                    for i in range(num_envs) if not dones[i]
                ])
                defender_actions = np.array([
                    defender_agent.select_action(observations['defender'][i], evaluate=True)
                    for i in range(num_envs) if not dones[i]
                ])
                
                # Pad actions for completed environments
                all_attacker_actions = np.zeros((num_envs, 1))
                all_defender_actions = np.zeros((num_envs, 1))
                active_idx = 0
                for i in range(num_envs):
                    if not dones[i]:
                        all_attacker_actions[i] = attacker_actions[active_idx]
                        all_defender_actions[i] = defender_actions[active_idx]
                        active_idx += 1
                
                # Step environments
                next_observations, _, new_dones, infos = eval_envs.step(all_attacker_actions, all_defender_actions)
                observations = next_observations
                
                # Process results for each environment
                for i in range(num_envs):
                    if not dones[i] and new_dones[i]:
                        dones[i] = True
                        episodes_done += 1
                        
                        # Record outcome
                        if infos[i]['terminal_condition'] in ['attacker_reached_target', 'unblocked_path_advantage']:
                            attacker_wins += 1
                        elif infos[i]['terminal_condition'] in ['defender_reached_target', 'attacker_captured', 'defender_time_advantage']:
                            defender_wins += 1
                        elif infos[i]['terminal_condition'] == 'timeout':
                            timeouts += 1
                        
                        # Update evaluation progress bar if enabled
                        if show_progress:
                            eval_pbar.update(1)
                            eval_pbar.set_description(f"Evaluating - A wins: {attacker_wins}, D wins: {defender_wins}")
                            
                        # Break if we've completed enough episodes
                        if episodes_done >= num_episodes:
                            break
            
            # Break if we've completed enough episodes
            if episodes_done >= num_episodes:
                break
        
        # Close the progress bar if enabled
        if show_progress:
            eval_pbar.close()
            
        # Calculate success rates
        episodes_completed = attacker_wins + defender_wins + timeouts
        attacker_success_rate = (attacker_wins / episodes_completed) * 100
        defender_success_rate = (defender_wins / episodes_completed) * 100
    
    else:
        # Single environment evaluation
        attacker_wins = 0
        defender_wins = 0
        timeouts = 0
        
        # Create evaluation progress bar if requested
        if show_progress:
            eval_pbar = tqdm(total=num_episodes, desc="Evaluating Agents")
            
        for _ in range(num_episodes):
            observations = env.reset()
            attacker_obs = observations['attacker']
            defender_obs = observations['defender']
            done = False
            
            while not done:
                # Use mean actions without exploration
                attacker_action = attacker_agent.select_action(attacker_obs, evaluate=True)
                defender_action = defender_agent.select_action(defender_obs, evaluate=True)
                
                next_observations, _, done, info = env.step(attacker_action, defender_action)
                attacker_obs = next_observations['attacker']
                defender_obs = next_observations['defender']
                
                # Check outcome once episode is done
                if done:
                    if info['terminal_condition'] in ['attacker_reached_target', 'unblocked_path_advantage']:
                        attacker_wins += 1
                    elif info['terminal_condition'] in ['defender_reached_target', 'attacker_captured', 'defender_time_advantage']:
                        defender_wins += 1
                    elif info['terminal_condition'] == 'timeout':
                        timeouts += 1
                    
                    # Update evaluation progress bar if enabled
                    if show_progress:
                        eval_pbar.update(1)
                        eval_pbar.set_description(f"Evaluating - A wins: {attacker_wins}, D wins: {defender_wins}")
        
        # Close the progress bar if enabled
        if show_progress:
            eval_pbar.close()
            
        # Calculate success rates
        attacker_success_rate = (attacker_wins / num_episodes) * 100
        defender_success_rate = (defender_wins / num_episodes) * 100
    
    return attacker_success_rate, defender_success_rate

def create_game_analytics_plot(env, episode, info, save_path):
    """Create an analytics plot showing trajectory and key metrics for both agents"""
    if not hasattr(env, 'trajectory_buffer') or len(env.trajectory_buffer['attacker_x']) < 2:
        return
        
    fig, axes = plt.subplots(2, 1, figsize=(12, 14))
    
    # Plot 1: Trajectory with agents' paths
    ax1 = axes[0]
    
    # Plot trajectories
    ax1.plot(
        env.trajectory_buffer['attacker_x'], 
        env.trajectory_buffer['attacker_y'], 
        'r-', linewidth=2, label='Attacker Path'
    )
    ax1.plot(
        env.trajectory_buffer['defender_x'], 
        env.trajectory_buffer['defender_y'], 
        'b-', linewidth=2, label='Defender Path'
    )
    
    # Plot start and end positions
    ax1.plot(env.trajectory_buffer['attacker_x'][0], env.trajectory_buffer['attacker_y'][0], 'ro', markersize=10, label='Attacker Start')
    ax1.plot(env.trajectory_buffer['attacker_x'][-1], env.trajectory_buffer['attacker_y'][-1], 'r*', markersize=10, label='Attacker End')
    ax1.plot(env.trajectory_buffer['defender_x'][0], env.trajectory_buffer['defender_y'][0], 'bo', markersize=10, label='Defender Start')
    ax1.plot(env.trajectory_buffer['defender_x'][-1], env.trajectory_buffer['defender_y'][-1], 'b*', markersize=10, label='Defender End')
    
    # Plot target
    ax1.plot(0, 0, 'g*', markersize=15, label='Target')
    
    # Plot defender capture radius at final position
    final_defender_circle = plt.Circle(
        (env.trajectory_buffer['defender_x'][-1], env.trajectory_buffer['defender_y'][-1]),
        env.r_capture, color='b', fill=False, alpha=0.3
    )
    ax1.add_patch(final_defender_circle)
    
    # Try to plot the Cartesian oval from the final state
    try:
        if hasattr(env, '_check_line_blocked'):
            env._check_line_blocked()  # This should update xsave, ysave
            
            if hasattr(env, 'xsave') and hasattr(env, 'ysave'):
                # Plot both branches of the Cartesian oval
                ax1.plot(env.xsave[:, 0], env.ysave[:, 0], 'r--', alpha=0.3, linewidth=1, label='Cartesian Oval')
                ax1.plot(env.xsave[:, 1], env.ysave[:, 1], 'r--', alpha=0.3, linewidth=1)
    except:
        pass  # If oval plotting fails, just skip it
    
    # Add labels and grid
    ax1.set_title(f"Episode {episode} Trajectory\nOutcome: {info['terminal_condition']}")
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    ax1.set_aspect('equal')
    
    # Plot 2: Metrics over time
    ax2 = axes[1]
    
    # Calculate metrics for each time step
    steps = range(len(env.trajectory_buffer['attacker_x']))
    distances_to_target = []
    distances_between_agents = []
    
    for i in steps:
        attacker_pos = np.array([env.trajectory_buffer['attacker_x'][i], env.trajectory_buffer['attacker_y'][i]], dtype=np.float32)
        defender_pos = np.array([env.trajectory_buffer['defender_x'][i], env.trajectory_buffer['defender_y'][i]], dtype=np.float32)
        target_pos = np.array([0.0, 0.0], dtype=np.float32)
        
        dist_to_target = np.linalg.norm(attacker_pos - target_pos)
        dist_between = np.linalg.norm(attacker_pos - defender_pos)
        
        distances_to_target.append(dist_to_target)
        distances_between_agents.append(dist_between)
    
    # Plot metrics
    ax2.plot(steps, distances_to_target, 'g-', label='Attacker to Target')
    ax2.plot(steps, distances_between_agents, 'm-', label='Attacker to Defender')
    
    # Add a horizontal line at capture radius
    ax2.axhline(y=env.r_capture, color='r', linestyle='--', label=f'Capture Radius ({env.r_capture})')
    
    # Add labels and grid
    ax2.set_title(f"Metrics Over Time\nAttacker Reward: {info['attacker_total_reward']:.2f}, Defender Reward: {info['defender_total_reward']:.2f}")
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Distance')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')
    
    # Add overall title
    fig.suptitle(f"Episode {episode} Analysis - {info['terminal_condition']}", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
    
    # Save figure
    fig.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close(fig)

#####################################################################
# MAIN IMPLEMENTATION
#####################################################################

def main():
    """
    Main function to set up and run dual-agent training for the differential game
    """
    # Start profiling
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Environment parameters
    env_params = {
        'gamma': 0.5,  # Defender speed relative to attacker
        'rho': 1.0,     # Capture radius parameter
        'dt': 0.1,      # Time step
        'render_debug': False,  # Disable visualization during training
        'normalize_rewards': True,
        'visualization_interval': 500  # Reduced visualization frequency
    }
    
    # Create the game environment
    env = DefenderAttackerGameEnv(**env_params)
    
    # Get environment dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = 1  # Both agents control their heading
    
    # Create attacker agent
    attacker_agent = SACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=256,
        actor_lr=3e-4,
        critic_lr=3e-4,
        gamma=0.99,
        tau=0.005,
        alpha=0.2
    )
    
    # Create defender agent
    defender_agent = SACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=256,
        actor_lr=3e-4,
        critic_lr=3e-4,
        gamma=0.99,
        tau=0.005,
        alpha=0.2
    )
    
    # Create replay buffers for both agents
    buffer_size = 1000000
    attacker_buffer = ReplayBuffer(
        capacity=buffer_size,
        state_dim=state_dim,
        action_dim=action_dim,
        seed=RANDOM_SEED
    )
    
    defender_buffer = ReplayBuffer(
        capacity=buffer_size,
        state_dim=state_dim,
        action_dim=action_dim,
        seed=RANDOM_SEED+1  # Different seed for diversity
    )
    
    # Training parameters
    num_episodes = 5000
    batch_size = 256
    updates_per_step = 1
    
    # Determine number of parallel environments to use
    num_envs = max(1, min(mp.cpu_count() - 1, 4))  # Use up to 4 environments or CPU count - 1
    
    # Start training
    print("Starting dual-agent training...")
    print(f"Using {num_envs} parallel environments")
    training_metrics = train_dual_agents(
        attacker_agent=attacker_agent,
        defender_agent=defender_agent,
        env=env,
        attacker_buffer=attacker_buffer,
        defender_buffer=defender_buffer,
        num_episodes=num_episodes,
        batch_size=batch_size,
        updates_per_step=updates_per_step,
        save_interval=100,
        eval_interval=100,
        vis_interval=500,  # Specify visualization interval
        use_parallel=True,
        num_envs=num_envs
    )
    
    # Disable profiler and save results
    profiler.disable()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    profiler.dump_stats(f'logs/profile_{timestamp}.prof')
    print(f"Training complete! Profiler results saved to logs/profile_{timestamp}.prof")
    print("You can analyze the profile with: python -m pstats logs/profile_{timestamp}.prof")
    
    # Final evaluation
    print("Running final evaluation...")
    attacker_success, defender_success = evaluate_dual_agents(
        attacker_agent, defender_agent, env, num_episodes=100)
    
    print(f"Final evaluation:")
    print(f"  Attacker Success Rate = {attacker_success:.2f}%")
    print(f"  Defender Success Rate = {defender_success:.2f}%")
    
    # Create visualizations of trained policies
    create_policy_visualizations(attacker_agent, defender_agent, env, num_episodes=5)

def create_policy_visualizations(attacker_agent, defender_agent, env, num_episodes=5, show_progress=True):
    """
    Create visualizations to analyze the learned policies of both agents
    """
    # Create directory for visualizations
    vis_dir = os.path.join('plots', 'final_policy_' + datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(vis_dir, exist_ok=True)
    
    # Ensure we're using a single environment for visualization
    if isinstance(env, ParallelEnvs):
        env = env.envs[0]  # Just use the first environment for visualization
    
    # Enable visualization
    env.set_visualization(True)
    env.render_debug = True
    
    # Create visualization progress bar if requested
    if show_progress:
        vis_pbar = tqdm(total=num_episodes, desc="Creating Visualizations")
    
    for episode in range(num_episodes):
        observations = env.reset()
        attacker_obs = observations['attacker']
        defender_obs = observations['defender']
        done = False
        
        while not done:
            # Use mean actions without exploration
            attacker_action = attacker_agent.select_action(attacker_obs, evaluate=True)
            defender_action = defender_agent.select_action(defender_obs, evaluate=True)
            
            next_observations, _, done, info = env.step(attacker_action, defender_action)
            attacker_obs = next_observations['attacker']
            defender_obs = next_observations['defender']
            
            # Render each step
            env.render()
            
            if done:
                # Save final frame
                fig_path = os.path.join(vis_dir, f"final_policy_episode_{episode}.png")
                if hasattr(env, 'fig') and env.fig is not None:
                    env.fig.savefig(fig_path, dpi=100, bbox_inches='tight')
                
                # Create analytics plot
                create_game_analytics_plot(env, episode, info, 
                                         os.path.join(vis_dir, f"final_analysis_{episode}.png"))
                
                # Update visualization progress bar if enabled
                if show_progress:
                    vis_pbar.update(1)
                    
                break
                
    # Close the visualization progress bar if enabled
    if show_progress:
        vis_pbar.close()
        
    print(f"Saved final policy visualizations to {vis_dir}")

if __name__ == "__main__":
    # Set PyTorch multiprocessing start method
    mp.set_start_method('spawn', force=True)
    main()