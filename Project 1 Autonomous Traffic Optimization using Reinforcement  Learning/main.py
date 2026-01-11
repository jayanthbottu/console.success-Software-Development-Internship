"""
Main Training Script for AI Traffic Control System
"""
import os
import argparse
from environment import TrafficEnvironment
from agent import DQNAgent
from gui import TrafficGUI
from analytics import TrainingAnalytics
from config import *

def train(render=True, load_model=None):
    """Main training loop"""
    
    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("analytics", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Initialize components
    env = TrafficEnvironment()
    agent = DQNAgent()
    gui = TrafficGUI() if render else None
    analytics = TrainingAnalytics()
    
    # Load existing model if specified
    if load_model and os.path.exists(load_model):
        agent.load_model(load_model)
        print(f"Continuing training from {load_model}")
    
    print("\n" + "="*60)
    print("AI TRAFFIC CONTROL - TRAINING MODE")
    print("="*60)
    print(f"Episodes: {NUM_EPISODES}")
    print(f"Max steps per episode: {MAX_STEPS_PER_EPISODE}")
    print(f"Rendering: {render}")
    print(f"Device: {agent.device}")
    print(f"Spawn rate: {SPAWN_RATE} (balanced across lanes)")
    print(f"Ambulance rate: {AMBULANCE_SPAWN_RATE}")
    print("="*60 + "\n")
    
    # Optional: Generate static controller baseline
    generate_baseline = input("Generate static controller baseline? (y/n) [n]: ").lower()
    if generate_baseline == 'y':
        analytics.simulate_static_controller(env, episodes=100)
    
    print("\nStarting training...")
    print("TIP: Watch for ambulances (red vehicles with cross)")
    print("TIP: Vehicles spawn in all 4 lanes now!\n")
    
    try:
        for episode in range(1, NUM_EPISODES + 1):
            state = env.reset()
            episode_reward = 0
            episode_losses = []
            wait_times = []
            
            for step in range(MAX_STEPS_PER_EPISODE):
                # Select and execute action
                action = agent.select_action(state, training=True)
                next_state, reward, done, info = env.step(action)
                
                # Store transition
                agent.store_transition(state, action, reward, next_state, done)
                
                # Train agent
                loss = agent.train_step()
                if loss is not None:
                    episode_losses.append(loss)
                
                # Update metrics
                episode_reward += reward
                wait_times.append(info['avg_wait'])
                state = next_state
                
                # Render if enabled
                if render and gui:
                    avg_loss = sum(episode_losses) / len(episode_losses) if episode_losses else None
                    gui.render(env, episode, agent.epsilon, avg_loss)
                    
                    if not gui.handle_events():
                        print("\nTraining interrupted by user")
                        raise KeyboardInterrupt
                
                if done:
                    break
            
            # Episode statistics
            avg_loss = sum(episode_losses) / len(episode_losses) if episode_losses else 0
            avg_wait = sum(wait_times) / len(wait_times) if wait_times else 0
            
            # Record analytics
            analytics.record_episode(
                episode, 
                episode_reward, 
                avg_loss, 
                avg_wait, 
                env.cars_passed, 
                agent.epsilon
            )
            
            # Print progress with ambulance info
            if episode % 5 == 0:
                amb_info = f"Amb: {env.ambulances_passed}" if env.ambulances_passed > 0 else ""
                print(f"Ep {episode:4d} | "
                      f"Reward: {episode_reward:7.1f} | "
                      f"Wait: {avg_wait:5.1f} | "
                      f"Cars: {env.cars_passed:3d} | {amb_info} | "
                      f"ε: {agent.epsilon:.3f} | "
                      f"Loss: {avg_loss:.4f}")
            
            # Save model periodically
            if episode % SAVE_INTERVAL == 0:
                agent.save_model(MODEL_SAVE_PATH)
                print(f"  → Model saved at episode {episode}")
    
    except KeyboardInterrupt:
        print("\n\nTraining interrupted!")
    
    finally:
        # Final save
        print("\nSaving final model...")
        agent.save_model(MODEL_SAVE_PATH)
        
        # Save analytics
        print("Saving analytics...")
        analytics.save_metrics()
        
        # Generate plots
        print("Generating performance plots...")
        analytics.plot_training_progress()
        analytics.print_summary()
        
        # Cleanup
        if gui:
            gui.close()
        
        print("\nTraining complete!")

def test(model_path, episodes=10):
    """Test trained model"""
    
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found!")
        return
    
    print("\n" + "="*60)
    print("AI TRAFFIC CONTROL - TEST MODE")
    print("="*60)
    print(f"Model: {model_path}")
    print(f"Test episodes: {episodes}")
    print("="*60 + "\n")
    
    # Initialize
    env = TrafficEnvironment()
    agent = DQNAgent()
    agent.load_model(model_path)
    agent.epsilon = 0.0  # No exploration during testing
    gui = TrafficGUI()
    
    total_rewards = []
    total_waits = []
    
    try:
        for episode in range(1, episodes + 1):
            state = env.reset()
            episode_reward = 0
            wait_times = []
            
            for step in range(MAX_STEPS_PER_EPISODE):
                action = agent.select_action(state, training=False)
                next_state, reward, done, info = env.step(action)
                
                episode_reward += reward
                wait_times.append(info['avg_wait'])
                state = next_state
                
                # Render
                gui.render(env, episode, 0.0, None)
                if not gui.handle_events():
                    raise KeyboardInterrupt
                
                if done:
                    break
            
            avg_wait = sum(wait_times) / len(wait_times) if wait_times else 0
            total_rewards.append(episode_reward)
            total_waits.append(avg_wait)
            
            print(f"Test Episode {episode}: "
                  f"Reward = {episode_reward:.1f}, "
                  f"Avg Wait = {avg_wait:.1f}, "
                  f"Cars Passed = {env.cars_passed}")
    
    except KeyboardInterrupt:
        print("\nTest interrupted")
    
    finally:
        gui.close()
        
        if total_rewards:
            print("\n" + "="*60)
            print("TEST RESULTS")
            print("="*60)
            print(f"Average Reward: {sum(total_rewards)/len(total_rewards):.2f}")
            print(f"Average Wait Time: {sum(total_waits)/len(total_waits):.2f}")
            print("="*60 + "\n")

def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description='AI Traffic Control System')
    parser.add_argument('--mode', choices=['train', 'test'], default='train',
                       help='Run mode: train or test')
    parser.add_argument('--no-render', action='store_true',
                       help='Disable visualization (faster training)')
    parser.add_argument('--load', type=str, default=None,
                       help='Load model from path')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of test episodes (test mode only)')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train(render=not args.no_render, load_model=args.load)
    elif args.mode == 'test':
        model_path = args.load if args.load else MODEL_SAVE_PATH
        test(model_path, episodes=args.episodes)

if __name__ == "__main__":
    main()