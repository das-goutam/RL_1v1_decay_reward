"""
Test script for the Defender-Attacker Differential Game
Loads the latest trained models and runs test episodes using the trained models
"""
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import time
import json
from datetime import datetime
import argparse
from tqdm import tqdm

# Import necessary components from the main training code
from train_1v1_decay_reward import (
    DefenderAttackerGameEnv, SACAgent, cartesian_oval, 
    create_game_analytics_plot, evaluate_dual_agents
)

class TestDefenderAttackerGameEnv(DefenderAttackerGameEnv):
    """Extend the environment to add the target_inside_oval terminal condition"""
    
    def step(self, attacker_action, defender_action):
        """Override step to add new terminal condition"""
        observations, rewards, done, info = super().step(attacker_action, defender_action)
        
        # Check if we should add the target_inside_oval terminal condition
        if not done and self.target_in_oval:
            # Calculate time to reach target for both agents
            distance_to_target = np.linalg.norm(self.xA - self.target)
            distance_defender_to_target = np.linalg.norm(self.xD - self.target)
            
            # Defender wins if target is inside oval
            terminal_reward_defender = distance_to_target * 1.5
            terminal_reward_attacker = -distance_to_target * 1.5
            done = True
            self.terminal_info = "target_inside_oval"
            
            # Update rewards
            rewards = {
                'attacker': terminal_reward_attacker,
                'defender': terminal_reward_defender
            }
            
            # Update totals
            self.attacker_total_reward += terminal_reward_attacker
            self.defender_total_reward += terminal_reward_defender
            self.done = done
            
            # Update info
            info["terminal_condition"] = self.terminal_info
            info["attacker_total_reward"] = self.attacker_total_reward
            info["defender_total_reward"] = self.defender_total_reward
            info["terminal_reward_attacker"] = terminal_reward_attacker
            info["terminal_reward_defender"] = terminal_reward_defender
            
            # Update observations
            observations = {
                'attacker': self._get_attacker_observation(),
                'defender': self._get_defender_observation()
            }
        
        return observations, rewards, done, info

def find_latest_model(model_prefix, model_dir='models'):
    """
    Find the latest model file matching the prefix in the model directory
    
    Args:
        model_prefix: Prefix for model files to search for (e.g., 'attacker_agent_')
        model_dir: Directory containing model files
        
    Returns:
        Path to the latest model file, or None if no models found
    """
    if not os.path.exists(model_dir):
        print(f"Model directory {model_dir} does not exist")
        return None
        
    model_files = [f for f in os.listdir(model_dir) if f.startswith(model_prefix) and f.endswith('.pt')]
    
    if not model_files:
        print(f"No models with prefix {model_prefix} found in {model_dir}")
        return None
    
    # Sort by modification time to get the latest model
    model_files.sort(key=lambda x: os.path.getmtime(os.path.join(model_dir, x)), reverse=True)
    latest_model_path = os.path.join(model_dir, model_files[0])
    
    return latest_model_path

def load_model_to_agent(agent, model_path):
    """
    Load a model to an agent with CPU mapping
    
    Args:
        agent: The agent to load the model into
        model_path: Path to the model file
        
    Returns:
        success: Whether the model was loaded successfully
    """
    try:
        # Try to load the model with CPU mapping
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        
        # Handle both DataParallel and non-DataParallel models
        if isinstance(agent.actor, torch.nn.DataParallel):
            # If current model is DataParallel but saved model is not
            if not any(k.startswith('module.') for k in checkpoint['actor']):
                # Convert checkpoint to DataParallel format
                for key in ['actor', 'q1', 'q2', 'q1_target', 'q2_target']:
                    checkpoint[key] = {'module.' + k: v for k, v in checkpoint[key].items()}
        else:
            # If current model is not DataParallel but saved model is
            if any(k.startswith('module.') for k in checkpoint['actor']):
                # Convert checkpoint to non-DataParallel format
                for key in ['actor', 'q1', 'q2', 'q1_target', 'q2_target']:
                    checkpoint[key] = {k.replace('module.', ''): v for k, v in checkpoint[key].items() 
                                     if k.startswith('module.')}
        
        # Load state dictionaries
        agent.actor.load_state_dict(checkpoint['actor'])
        agent.q1.load_state_dict(checkpoint['q1'])
        agent.q2.load_state_dict(checkpoint['q2'])
        agent.q1_target.load_state_dict(checkpoint['q1_target'])
        agent.q2_target.load_state_dict(checkpoint['q2_target'])
        
        # Load optimizer state dictionaries if they exist and we're not in eval mode
        if 'actor_optimizer' in checkpoint:
            try:
                agent.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
                agent.q1_optimizer.load_state_dict(checkpoint['q1_optimizer'])
                agent.q2_optimizer.load_state_dict(checkpoint['q2_optimizer'])
            except Exception as e:
                print(f"Could not load optimizer states: {e}")
                
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def run_test_episodes(env, attacker_agent, defender_agent, num_episodes=5, render=True, save_dir=None, show_progress=True):
    """
    Run test episodes using the provided agents and environment
    
    Args:
        env: The game environment
        attacker_agent: The trained attacker agent
        defender_agent: The trained defender agent
        num_episodes: Number of episodes to run
        render: Whether to render the episodes
        save_dir: Directory to save visualizations (if None, no saving)
        show_progress: Whether to show a progress bar
    
    Returns:
        results: Dictionary with episode results
    """
    # Create save directory if needed
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Results tracking
    results = {
        'attacker_wins': 0,
        'defender_wins': 0,
        'timeouts': 0,
        'episode_lengths': [],
        'terminal_conditions': {},
        'episode_data': []
    }
    
    # Enable visualization for all episodes if rendering
    if render:
        env.set_visualization(True)
        env.render_debug = True
    
    # Create progress bar if requested
    if show_progress:
        episodes_iter = tqdm(range(num_episodes), desc="Testing Agents")
    else:
        episodes_iter = range(num_episodes)
    
    # Run episodes
    for episode in episodes_iter:
        # Reset environment and get initial observations
        observations = env.reset()
        attacker_obs = observations['attacker']
        defender_obs = observations['defender']
        
        # Episode loop
        done = False
        step = 0
        
        while not done:
            # Use mean actions without exploration for testing
            attacker_action = attacker_agent.select_action(attacker_obs, evaluate=True)
            defender_action = defender_agent.select_action(defender_obs, evaluate=True)
            
            # Take step in environment
            next_observations, rewards, done, info = env.step(attacker_action, defender_action)
            
            # Update observations
            attacker_obs = next_observations['attacker']
            defender_obs = next_observations['defender']
            
            # Render if enabled
            if render:
                env.render()
                plt.pause(0.02)  # Slightly longer pause for better visualization
            
            step += 1
            
            # Break if episode is done
            if done:
                # Record outcome
                terminal_condition = info.get('terminal_condition', 'unknown')
                if terminal_condition not in results['terminal_conditions']:
                    results['terminal_conditions'][terminal_condition] = 0
                results['terminal_conditions'][terminal_condition] += 1
                
                # Record win/loss
                if terminal_condition in ['attacker_reached_target', 'unblocked_path_advantage']:
                    results['attacker_wins'] += 1
                    outcome = "Attacker Won"
                elif terminal_condition in ['defender_reached_target', 'attacker_captured', 'defender_time_advantage', 'target_inside_oval']:
                    results['defender_wins'] += 1
                    outcome = "Defender Won"
                else:
                    results['timeouts'] += 1
                    outcome = "Timeout"
                
                # Record episode length
                results['episode_lengths'].append(step)
                
                # Record episode data
                episode_data = {
                    'episode': episode + 1,
                    'steps': step,
                    'outcome': outcome,
                    'terminal_condition': terminal_condition,
                    'attacker_reward': info['attacker_total_reward'],
                    'defender_reward': info['defender_total_reward'],
                    'target_in_oval': info['target_in_oval'],
                    'distance_to_target': info['distance_to_target'],
                    'defender_to_target': info['defender_to_target'],
                    'DA': info['DA']
                }
                results['episode_data'].append(episode_data)
                
                # Update progress bar if enabled
                if show_progress:
                    episodes_iter.set_description(
                        f"Testing - A wins: {results['attacker_wins']}, D wins: {results['defender_wins']}, "
                        f"Timeouts: {results['timeouts']}, Avg Steps: {np.mean(results['episode_lengths']):.1f}"
                    )
                
                # Save visualizations if enabled
                if save_dir and hasattr(env, 'fig') and env.fig is not None:
                    # Save episode visualization
                    fig_path = os.path.join(save_dir, f"test_episode_{episode+1}.png")
                    env.fig.savefig(fig_path, dpi=100, bbox_inches='tight')
                    
                    # Create and save analytics plot
                    analytics_path = os.path.join(save_dir, f"analysis_episode_{episode+1}.png")
                    create_game_analytics_plot(env, episode+1, info, analytics_path)
                
                break
    
    # Calculate aggregate statistics
    results['avg_episode_length'] = np.mean(results['episode_lengths'])
    results['attacker_win_rate'] = results['attacker_wins'] / num_episodes * 100
    results['defender_win_rate'] = results['defender_wins'] / num_episodes * 100
    results['timeout_rate'] = results['timeouts'] / num_episodes * 100
    
    # Print overall results
    print("\nTest Results Summary:")
    print(f"  Episodes run: {num_episodes}")
    print(f"  Attacker wins: {results['attacker_wins']} ({results['attacker_win_rate']:.1f}%)")
    print(f"  Defender wins: {results['defender_wins']} ({results['defender_win_rate']:.1f}%)")
    print(f"  Timeouts: {results['timeouts']} ({results['timeout_rate']:.1f}%)")
    print(f"  Average episode length: {results['avg_episode_length']:.1f} steps")
    print("\nTerminal conditions:")
    for condition, count in results['terminal_conditions'].items():
        print(f"  {condition}: {count} ({count/num_episodes*100:.1f}%)")
    
    # Save results to JSON if save_dir is provided
    if save_dir:
        results_path = os.path.join(save_dir, "test_results.json")
        with open(results_path, 'w') as f:
            # Convert numpy values to Python native types for JSON serialization
            json_results = {
                k: v if not isinstance(v, np.ndarray) and not isinstance(v, np.generic) 
                else v.tolist() if isinstance(v, np.ndarray) 
                else float(v) 
                for k, v in results.items()
            }
            json.dump(json_results, f, indent=2)
        print(f"Results saved to {results_path}")
    
    return results

def main():
    """Main function to run test episodes"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test Defender-Attacker Differential Game')
    parser.add_argument('--model_dir', type=str, default='models', help='Directory containing model files')
    parser.add_argument('--episodes', type=int, default=5, help='Number of test episodes to run')
    parser.add_argument('--render', action='store_true', help='Enable rendering')
    parser.add_argument('--no-render', dest='render', action='store_false', help='Disable rendering')
    parser.add_argument('--gamma', type=float, default=0.5, help='Defender speed relative to attacker')
    parser.add_argument('--rho', type=float, default=1.0, help='Capture radius parameter')
    parser.set_defaults(render=True)
    args = parser.parse_args()
    
    print("Starting test of Defender-Attacker Differential Game")
    print("With added terminal condition: target_inside_oval")
    
    # Environment parameters
    env_params = {
        'gamma': args.gamma,
        'rho': args.rho,
        'dt': 0.1,
        'render_debug': args.render,
        'normalize_rewards': True,
        'visualization_interval': 1  # Visualize all episodes
    }
    
    # Create the modified environment with target_inside_oval terminal condition
    env = TestDefenderAttackerGameEnv(**env_params)
    print(f"Environment created with gamma={args.gamma}, rho={args.rho}")
    
    # Get environment dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = 1  # Both agents control their heading
    
    # Find latest model files
    attacker_model_path = find_latest_model('attacker_agent_', args.model_dir)
    defender_model_path = find_latest_model('defender_agent_', args.model_dir)
    
    if not attacker_model_path or not defender_model_path:
        print("Could not find model files. Exiting.")
        return
    
    # Create agents
    attacker_agent = SACAgent(state_dim, action_dim)
    defender_agent = SACAgent(state_dim, action_dim)
    
    # Load models with CPU mapping
    print(f"Loading attacker model: {attacker_model_path}")
    attacker_success = load_model_to_agent(attacker_agent, attacker_model_path)
    
    print(f"Loading defender model: {defender_model_path}")
    defender_success = load_model_to_agent(defender_agent, defender_model_path)
    
    if not attacker_success or not defender_success:
        print("Failed to load one or both models. Exiting.")
        return
    
    print("Models loaded successfully!")
    
    # Create save directory for test visualizations
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join('test_results', timestamp)
    
    # Run test episodes
    print(f"\nRunning {args.episodes} test episodes...")
    
    results = run_test_episodes(
        env=env,
        attacker_agent=attacker_agent,
        defender_agent=defender_agent,
        num_episodes=args.episodes,
        render=args.render,
        save_dir=save_dir,
        show_progress=True
    )
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    main()