# 🚦 AI Traffic Control System - Deep Reinforcement Learning

A sophisticated traffic signal optimization system using Deep Q-Network (DQN) to minimize vehicle waiting times and prioritize emergency vehicles.

## 🎯 Project Overview

This project implements an intelligent traffic control system that learns optimal signal timing through reinforcement learning. Unlike traditional static timer-based systems, our AI agent adapts to real-time traffic conditions and emergency vehicle priorities.

### Key Features

- **Deep Q-Network (DQN)** with experience replay and target networks
- **Adaptive signal control** based on real-time vehicle density
- **Emergency vehicle prioritization** - Immediate green light for ambulances
- **Comprehensive analytics** - AI vs Static controller comparison
- **Real-time visualization** - PyGame-based GUI with live metrics
- **Professional architecture** - Modular, extensible, production-ready code

## 📁 Project Structure

```
ai-traffic-control/
├── config.py              # All hyperparameters and constants
├── vehicle.py             # Vehicle class and movement logic
├── environment.py         # Traffic environment (RL interface)
├── agent.py               # DQN agent implementation
├── gui.py                 # Visualization and rendering
├── analytics.py           # Performance tracking and plotting
├── main.py                # Training/testing entry point
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── models/               # Saved model checkpoints
├── analytics/            # Performance plots and metrics
└── logs/                 # Training logs
```

## 🚀 Installation

### Prerequisites
- Python 3.8+
- CUDA (optional, for GPU acceleration)

### Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd ai-traffic-control

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```txt
torch>=2.0.0
numpy>=1.24.0
pygame>=2.5.0
matplotlib>=3.7.0
```

## 💻 Usage

### Training Mode

Train the AI agent from scratch:
```bash
python main.py --mode train
```

Train without visualization (faster):
```bash
python main.py --mode train --no-render
```

Continue training from checkpoint:
```bash
python main.py --mode train --load models/traffic_dqn.pth
```

### Testing Mode

Test a trained model:
```bash
python main.py --mode test --load models/traffic_dqn.pth
```

Run custom number of test episodes:
```bash
python main.py --mode test --load models/traffic_dqn.pth --episodes 20
```

## 🧠 Reinforcement Learning Design

### State Space (16 dimensions)
- **Vehicle counts per lane** (4) - Normalized 0-1
- **Average wait times per lane** (4) - Normalized 0-1
- **Current green light** (4) - One-hot encoded
- **Ambulance presence per lane** (4) - Binary flags

### Action Space (2 actions)
- **0**: Maintain current green light
- **1**: Switch to next lane (triggers yellow phase)

### Reward Function

```python
Reward = -1.5 × waiting_cars 
         -3.0 × switch_action
         +150 × ambulance_moving
         -100 × ambulance_waiting
         -0.5 × congestion_penalty
```

**Reward Philosophy:**
- Minimize total waiting time
- Penalize unnecessary switching (encourage stability)
- Heavily prioritize ambulance movement
- Penalize congestion beyond threshold

### Network Architecture

```
Input (16) → FC(128) → ReLU → Dropout(0.2)
           → FC(128) → ReLU → Dropout(0.2)
           → FC(64)  → ReLU
           → Output(2)
```

### Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Learning Rate | 0.0005 | Adam optimizer |
| Batch Size | 128 | Experience replay |
| Gamma (γ) | 0.97 | Discount factor |
| Memory Size | 10,000 | Replay buffer |
| Epsilon Start | 1.0 | Initial exploration |
| Epsilon End | 0.05 | Minimum exploration |
| Epsilon Decay | 0.9965 | Per-episode decay |
| Target Update | 500 steps | Hard update frequency |

## 📊 Performance Metrics

The system tracks and compares:

1. **Average Wait Time** - AI vs Static controller
2. **Throughput** - Vehicles passed per episode
3. **Training Loss** - DQN loss convergence
4. **Epsilon Decay** - Exploration vs exploitation
5. **Emergency Response** - Ambulance waiting times

### Expected Results

After ~500 episodes of training:
- **30-50% reduction** in average wait times vs static controller
- **15-25% improvement** in throughput
- **Near-instant** ambulance prioritization
- Stable convergence with epsilon < 0.1

## 🎮 GUI Controls

- **ESC** - Exit simulation
- **Close Window** - Stop training/testing

### Dashboard Information

- **Traffic Lights** - Current signal states per lane
- **Training Stats** - Episode, score, epsilon, loss
- **Vehicle Counts** - Real-time per-lane counts
- **Ambulance Alert** - Flashing indicator when present

## 🔧 Configuration

Edit `config.py` to customize:

```python
# Training
NUM_EPISODES = 800
MAX_STEPS_PER_EPISODE = 1500

# Environment
SPAWN_RATE = 0.03  # Vehicle spawn probability
AMBULANCE_SPAWN_RATE = 0.008

# RL Parameters
LEARNING_RATE = 0.0005
BATCH_SIZE = 128
GAMMA = 0.97

# Traffic Rules
MIN_GREEN_TIME = 40  # Minimum frames before switch
YELLOW_DURATION = 90  # Yellow light duration
```

## 📈 Analytics Output

After training, the system generates:

### Plots (`analytics/training_progress_*.png`)
1. Episode Rewards (with moving average)
2. Training Loss over time
3. Epsilon decay curve
4. Wait Time: AI vs Static comparison
5. Throughput: AI vs Static comparison
6. Learning progress (first 50 vs last 50 episodes)

### Metrics (`analytics/metrics_*.json`)
- Complete training history
- Statistical summary
- AI vs Static comparison data

## 🏗️ Architecture Highlights

### Modular Design
- **Separation of concerns** - Each module handles one responsibility
- **Easy to extend** - Add new vehicle types, reward functions, or RL algorithms
- **Configuration-driven** - All parameters in one place

### Professional Practices
- **Type hints** - Clear function signatures
- **Docstrings** - Comprehensive documentation
- **Error handling** - Graceful degradation
- **Logging** - Training progress tracking

### Optimization
- **GPU acceleration** - Automatic CUDA detection
- **Vectorized operations** - NumPy/PyTorch efficiency
- **Experience replay** - Sample efficiency
- **Target network** - Training stability

## 🔬 Advanced Features

### Double DQN (Optional Enhancement)
To implement Double DQN, modify `agent.py`:
```python
# In train_step(), use policy network for action selection
next_actions = self.policy_net(next_states).argmax(1).unsqueeze(1)
next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze()
```

### Prioritized Experience Replay
Add priority weights to replay buffer for faster learning on important transitions.

### Multi-Intersection
Extend environment to handle multiple connected intersections with shared state.

## 🐛 Troubleshooting

**Issue**: Slow training
- **Solution**: Run with `--no-render` flag
- **Solution**: Reduce `MAX_STEPS_PER_EPISODE` in config

**Issue**: Poor performance after training
- **Solution**: Train longer (1000+ episodes)
- **Solution**: Adjust reward function weights
- **Solution**: Increase `MIN_GREEN_TIME` to reduce switching

**Issue**: GPU not detected
- **Solution**: Install CUDA-enabled PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/cu118`

## 📝 Future Enhancements

- [ ] Multi-intersection coordination
- [ ] Pedestrian crossing logic
- [ ] Time-of-day traffic patterns
- [ ] Weather conditions simulation
- [ ] Real traffic data integration (SUMO)
- [ ] Web-based dashboard
- [ ] Model deployment on edge devices

## 🤝 Contributing

This is an academic project, but contributions are welcome:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Submit a pull request

## 📄 License

MIT License - See LICENSE file for details

## 🎓 Academic Context

**Project**: Autonomous Traffic Optimization using Reinforcement Learning  
**Objective**: Demonstrate AI superiority over traditional static controllers  
**Key Learning**: Deep RL, traffic simulation, emergency prioritization, performance analysis

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Built with** ❤️ **using PyTorch, NumPy, and PyGame**