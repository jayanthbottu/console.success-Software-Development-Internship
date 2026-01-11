"""
Analytics Module - Track and visualize training performance
"""
import matplotlib.pyplot as plt
import numpy as np
import json
import os
from datetime import datetime

class TrainingAnalytics:
    """Track and visualize training metrics"""
    
    def __init__(self, save_dir="analytics"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Training metrics
        self.episode_rewards = []
        self.episode_losses = []
        self.episode_wait_times = []
        self.episode_throughput = []
        self.epsilon_history = []
        self.ambulance_response_times = []
        
        # Comparison data for static controller
        self.static_wait_times = []
        self.static_throughput = []
    
    def record_episode(self, episode, reward, avg_loss, avg_wait, throughput, epsilon, amb_response=None):
        """Record metrics for an episode"""
        self.episode_rewards.append(reward)
        self.episode_losses.append(avg_loss if avg_loss is not None else 0)
        self.episode_wait_times.append(avg_wait)
        self.episode_throughput.append(throughput)
        self.epsilon_history.append(epsilon)
        
        if amb_response is not None:
            self.ambulance_response_times.append(amb_response)
    
    def simulate_static_controller(self, env, episodes=100):
        """
        Simulate a static timer-based controller for comparison
        Fixed green time: 150 frames per direction
        """
        print("\nSimulating static timer controller for comparison...")
        
        for ep in range(episodes):
            env.reset()
            total_wait = 0
            measurements = 0
            
            for step in range(1500):
                # Static switching every 150 steps
                if step % 150 == 0:
                    env.current_green = (env.current_green + 1) % 4
                
                _, _, _, info = env.step(0)  # Always "stay" action
                total_wait += info['avg_wait']
                measurements += 1
            
            avg_wait = total_wait / measurements if measurements > 0 else 0
            self.static_wait_times.append(avg_wait)
            self.static_throughput.append(env.cars_passed)
            
            if (ep + 1) % 20 == 0:
                print(f"  Static simulation {ep+1}/{episodes}")
        
        print(f"Static controller avg wait time: {np.mean(self.static_wait_times):.2f}")
        print(f"Static controller avg throughput: {np.mean(self.static_throughput):.2f}")
    
    def plot_training_progress(self):
        """Generate comprehensive training plots"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('AI Traffic Control - Training Performance', fontsize=16, fontweight='bold')
        
        episodes = range(1, len(self.episode_rewards) + 1)
        
        # 1. Episode Rewards
        ax = axes[0, 0]
        ax.plot(episodes, self.episode_rewards, alpha=0.6, color='blue', linewidth=0.8)
        if len(self.episode_rewards) > 20:
            smoothed = self._smooth(self.episode_rewards, 20)
            ax.plot(episodes, smoothed, color='darkblue', linewidth=2, label='Moving Avg (20)')
            ax.legend()
        ax.set_xlabel('Episode')
        ax.set_ylabel('Total Reward')
        ax.set_title('Episode Rewards')
        ax.grid(True, alpha=0.3)
        
        # 2. Training Loss
        ax = axes[0, 1]
        ax.plot(episodes, self.episode_losses, alpha=0.6, color='red', linewidth=0.8)
        if len(self.episode_losses) > 20:
            smoothed = self._smooth(self.episode_losses, 20)
            ax.plot(episodes, smoothed, color='darkred', linewidth=2, label='Moving Avg (20)')
            ax.legend()
        ax.set_xlabel('Episode')
        ax.set_ylabel('Average Loss')
        ax.set_title('Training Loss')
        ax.grid(True, alpha=0.3)
        
        # 3. Epsilon Decay
        ax = axes[0, 2]
        ax.plot(episodes, self.epsilon_history, color='green', linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Epsilon')
        ax.set_title('Exploration Rate (Epsilon)')
        ax.grid(True, alpha=0.3)
        
        # 4. Average Wait Time Comparison
        ax = axes[1, 0]
        ax.plot(episodes, self.episode_wait_times, alpha=0.6, color='orange', 
                linewidth=0.8, label='AI Controller')
        if len(self.episode_wait_times) > 20:
            smoothed = self._smooth(self.episode_wait_times, 20)
            ax.plot(episodes, smoothed, color='darkorange', linewidth=2, label='AI Moving Avg')
        
        if self.static_wait_times:
            static_avg = np.mean(self.static_wait_times)
            ax.axhline(y=static_avg, color='gray', linestyle='--', linewidth=2, 
                      label=f'Static Controller ({static_avg:.1f})')
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Average Wait Time (frames)')
        ax.set_title('Wait Time: AI vs Static Controller')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. Throughput Comparison
        ax = axes[1, 1]
        ax.plot(episodes, self.episode_throughput, alpha=0.6, color='purple', 
                linewidth=0.8, label='AI Controller')
        if len(self.episode_throughput) > 20:
            smoothed = self._smooth(self.episode_throughput, 20)
            ax.plot(episodes, smoothed, color='darkviolet', linewidth=2, label='AI Moving Avg')
        
        if self.static_throughput:
            static_avg = np.mean(self.static_throughput)
            ax.axhline(y=static_avg, color='gray', linestyle='--', linewidth=2,
                      label=f'Static Controller ({static_avg:.0f})')
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Cars Passed')
        ax.set_title('Throughput: AI vs Static Controller')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 6. Performance Improvement
        ax = axes[1, 2]
        if len(self.episode_wait_times) > 50:
            first_50 = np.mean(self.episode_wait_times[:50])
            last_50 = np.mean(self.episode_wait_times[-50:])
            improvement = ((first_50 - last_50) / first_50) * 100
            
            bars = ax.bar(['First 50\nEpisodes', 'Last 50\nEpisodes'], 
                         [first_50, last_50], 
                         color=['lightcoral', 'lightgreen'])
            ax.set_ylabel('Average Wait Time')
            ax.set_title(f'Learning Progress\n({improvement:.1f}% improvement)')
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}', ha='center', va='bottom')
        else:
            ax.text(0.5, 0.5, 'Insufficient data\n(need 50+ episodes)', 
                   ha='center', va='center', transform=ax.transAxes)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.save_dir}/training_progress_{timestamp}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"\nTraining plots saved to: {filename}")
        
        plt.show()
    
    def _smooth(self, data, window):
        """Apply moving average smoothing"""
        return np.convolve(data, np.ones(window)/window, mode='same')
    
    def save_metrics(self):
        """Save metrics to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.save_dir}/metrics_{timestamp}.json"
        
        data = {
            'episode_rewards': self.episode_rewards,
            'episode_losses': self.episode_losses,
            'episode_wait_times': self.episode_wait_times,
            'episode_throughput': self.episode_throughput,
            'epsilon_history': self.epsilon_history,
            'static_wait_times': self.static_wait_times,
            'static_throughput': self.static_throughput,
            'summary': {
                'total_episodes': len(self.episode_rewards),
                'final_avg_wait': np.mean(self.episode_wait_times[-50:]) if len(self.episode_wait_times) >= 50 else 0,
                'final_avg_throughput': np.mean(self.episode_throughput[-50:]) if len(self.episode_throughput) >= 50 else 0,
                'static_avg_wait': np.mean(self.static_wait_times) if self.static_wait_times else 0,
                'static_avg_throughput': np.mean(self.static_throughput) if self.static_throughput else 0
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Metrics saved to: {filename}")
    
    def print_summary(self):
        """Print training summary"""
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        
        if len(self.episode_wait_times) >= 50:
            first_50_wait = np.mean(self.episode_wait_times[:50])
            last_50_wait = np.mean(self.episode_wait_times[-50:])
            wait_improvement = ((first_50_wait - last_50_wait) / first_50_wait) * 100
            
            print(f"\nWait Time Performance:")
            print(f"  First 50 episodes: {first_50_wait:.2f} frames")
            print(f"  Last 50 episodes:  {last_50_wait:.2f} frames")
            print(f"  Improvement:       {wait_improvement:.1f}%")
        
        if self.static_wait_times:
            ai_wait = np.mean(self.episode_wait_times[-100:]) if len(self.episode_wait_times) >= 100 else np.mean(self.episode_wait_times)
            static_wait = np.mean(self.static_wait_times)
            vs_static = ((static_wait - ai_wait) / static_wait) * 100
            
            print(f"\nAI vs Static Controller:")
            print(f"  AI avg wait time:     {ai_wait:.2f} frames")
            print(f"  Static avg wait time: {static_wait:.2f} frames")
            print(f"  AI improvement:       {vs_static:.1f}%")
        
        if self.episode_throughput:
            avg_throughput = np.mean(self.episode_throughput[-50:]) if len(self.episode_throughput) >= 50 else np.mean(self.episode_throughput)
            print(f"\nThroughput:")
            print(f"  Average cars passed: {avg_throughput:.1f} per episode")
        
        print("\n" + "="*60)