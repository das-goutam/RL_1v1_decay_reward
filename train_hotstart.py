"""
Defender-Attacker Game Differential Game Training with Hot-start Capability
This version allows continuing training from previously trained models
"""

import os
import argparse
import torch
import numpy as np
from datetime import datetime

# Import everything from your main code
from train_1v1_decay_reward import (
    DefenderAttackerGameEnv, SACAgent, ReplayBuffer, cartesian_oval,
    train_dual_agents, evaluate_dual_agents, RANDOM_SEED
)

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
    
    # Extract the episode number if possible (for reporting)
    episode_num = None
    try:
        # Try to find episode number in filename (e.g., attacker_agent_20240304_episode_2000.pt)
        parts = model_files[0].split('_')
        for i, part in enumerate(parts):
            if part == 'episode' and i < len(parts) - 1:
                episode_num = int(parts[i+1])
                break
    except:
        pass
    
    return latest_model_path, episode_num

def load_model_to_agent(agent, model_path):
    """
    Load a model to an agent with CPU mapping if needed
    
    Args:
        agent: The agent to load the model into
        model_path: Path to the model file
        
    Returns:
        success: Whether the model was loaded successfully
    """
    try:
        # Try to load the model with CPU mapping if CUDA isn't available
        map_location = None if torch.cuda.is_available() else torch.device('cpu')
        checkpoint = torch.load(model_path, map_location=map_location)
        
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
                    new_state_dict = {}
                    for k, v in checkpoint[key].items():
                        if k.startswith('module.'):
                            new_state_dict[k[7:]] = v  # Remove 'module.' prefix
                        else:
                            new_state_dict[k] = v
                    checkpoint[key] = new_state_dict
        
        # Load state dictionaries
        agent.actor.load_state_dict(checkpoint['actor'])
        agent.q1.load_state_dict(checkpoint['q1'])
        agent.q2.load_state_dict(checkpoint['q2'])
        agent.q1_target.load_state_dict(checkpoint['q1_target'])
        agent.q2_target.load_state_dict(checkpoint['q2_target'])
        
        # Load optimizer state dictionaries if they exist
        if 'actor_optimizer' in checkpoint:
            try:
                agent.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
                agent.q1_optimizer.load_state_dict(checkpoint['q1_optimizer'])
                agent.q2_optimizer.load_state_dict(checkpoint['q2_optimizer'])
            except Exception as e:
                print(f"Could not load optimizer states, using default initialization. Error: {e}")
                
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def main():
    """Main function to set up and run dual-agent training with hot-start capability"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Defender-Attacker Game with Hot-start')
    parser.add_argument('--episodes', type=int, default=5000, help='Number of episodes to train')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--buffer_size', type=int, default=1000000, help='Replay buffer size')
    parser.add_argument('--eval_interval', type=int, default=100, help='Evaluation interval')
    parser.add_argument('--save_interval', type=int, default=100, help='Model save interval')
    parser.add_argument('--vis_interval', type=int, default=500, help='Visualization interval')
    parser.add_argument('--model_dir', type=str, default='models', help='Directory for models')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory for logs')
    parser.add_argument('--num_envs', type=int, default=4, help='Number of parallel environments')
    parser.add_argument('--hotstart', action='store_true', help='Enable hot-start from latest models')
    parser.add_argument('--no-hotstart', dest='hotstart', action='store_false', help='Disable hot-start')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED, help='Random seed')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')
    parser.set_defaults(hotstart=True)  # Default to using hot-start
    
    args = parser.parse_args()
    
    # Set GPU device if specified
    if torch.cuda.is_available() and args.gpu >= 0:
        torch.cuda.set_device(args.gpu)
        print(f"Using GPU: {torch.cuda.get_device_name(args.gpu)}")
    
    # Set random seed
    if args.seed != RANDOM_SEED:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
    
    # Environment parameters
    env_params = {
        'gamma': 0.5,  # Defender speed relative to attacker
        'rho': 1.0,    # Capture radius parameter
        'dt': 0.1,     # Time step
        'render_debug': False,  # Disable visualization during training
        'normalize_rewards': True,
        'visualization_interval': args.vis_interval
    }
    
    # Create the game environment
    env = DefenderAttackerGameEnv(**env_params)
    
    # Get environment dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = 1  # Both agents control their heading
    
    # Create agents
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
    
    # Create timestamp for this training run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Hot-start from previous models if requested
    starting_episode = 0
    
    if args.hotstart:
        print("Looking for latest models to hot-start training...")
        
        # Find latest attacker and defender models
        attacker_model_path, attacker_episode = find_latest_model('attacker_agent_', args.model_dir)
        defender_model_path, defender_episode = find_latest_model('defender_agent_', args.model_dir)
        
        if attacker_model_path and defender_model_path:
            print(f"Found latest attacker model: {attacker_model_path}")
            print(f"Found latest defender model: {defender_model_path}")
            
            # Load models
            attacker_success = load_model_to_agent(attacker_agent, attacker_model_path)
            defender_success = load_model_to_agent(defender_agent, defender_model_path)
            
            if attacker_success and defender_success:
                print("Successfully loaded both models for hot-start training!")
                
                # Set starting episode if available
                if attacker_episode is not None and defender_episode is not None:
                    # Use the minimum of both episodes to be safe
                    starting_episode = min(attacker_episode, defender_episode)
                    print(f"Continuing from episode {starting_episode}")
            else:
                print("Failed to load one or both models. Starting training from scratch.")
        else:
            print("No existing models found. Starting training from scratch.")
    else:
        print("Hot-start disabled. Starting training from scratch.")
    
    # Create replay buffers for both agents
    attacker_buffer = ReplayBuffer(
        capacity=args.buffer_size,
        state_dim=state_dim,
        action_dim=action_dim,
        seed=args.seed
    )
    
    defender_buffer = ReplayBuffer(
        capacity=args.buffer_size,
        state_dim=state_dim,
        action_dim=action_dim,
        seed=args.seed+1  # Different seed for diversity
    )
    
    # Determine number of parallel environments to use
    import multiprocessing as mp
    num_envs = min(args.num_envs, mp.cpu_count() - 1)  # Don't use more than CPU count - 1
    num_envs = max(1, num_envs)  # Use at least 1
    
    # Calculate remaining episodes to train
    remaining_episodes = max(0, args.episodes - starting_episode)
    
    if remaining_episodes <= 0:
        print(f"Already trained for {starting_episode} episodes, which is >= the requested {args.episodes}.")
        print("Nothing to do. If you want to train more, increase the --episodes parameter.")
        return
    
    # Start training
    print(f"Starting dual-agent training for {remaining_episodes} more episodes...")
    print(f"Using {num_envs} parallel environments")
    
    training_metrics = train_dual_agents(
        attacker_agent=attacker_agent,
        defender_agent=defender_agent,
        env=env,
        attacker_buffer=attacker_buffer,
        defender_buffer=defender_buffer,
        num_episodes=remaining_episodes,  # Train for the remaining episodes
        batch_size=args.batch_size,
        updates_per_step=1,
        save_interval=args.save_interval,
        eval_interval=args.eval_interval,
        vis_interval=args.vis_interval,
        model_dir=args.model_dir,
        log_dir=args.log_dir,
        use_parallel=True,
        num_envs=num_envs
    )
    
    # Final evaluation
    print("Running final evaluation...")
    attacker_success, defender_success = evaluate_dual_agents(
        attacker_agent, defender_agent, env, num_episodes=100)
    
    print(f"Final evaluation:")
    print(f"  Attacker Success Rate = {attacker_success:.2f}%")
    print(f"  Defender Success Rate = {defender_success:.2f}%")
    
    print(f"Training completed successfully!")
    print(f"Total episodes trained: {starting_episode + remaining_episodes}")

if __name__ == "__main__":
    # Set PyTorch multiprocessing start method
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    main()