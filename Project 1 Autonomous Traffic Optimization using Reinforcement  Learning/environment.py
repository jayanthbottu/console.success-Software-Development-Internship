"""
Traffic Environment - Manages simulation state and RL interface
"""
import numpy as np
import random
from vehicle import Vehicle
from config import *

class TrafficEnvironment:
    """OpenAI Gym-style environment for traffic control"""
    
    def __init__(self):
        self.vehicles = []
        self.current_green = 0
        self.time_in_phase = 0
        self.is_yellow = False
        self.yellow_timer = 0
        
        # Statistics
        self.total_steps = 0
        self.cars_passed = 0
        self.ambulances_passed = 0
        self.episode_reward = 0
        
        # Per-lane metrics
        self.lane_wait_times = [0, 0, 0, 0]
        self.lane_counts = [0, 0, 0, 0]
    
    def reset(self):
        """Reset environment to initial state"""
        self.vehicles = []
        self.current_green = 0
        self.time_in_phase = 0
        self.is_yellow = False
        self.yellow_timer = 0
        self.total_steps = 0
        self.cars_passed = 0
        self.ambulances_passed = 0
        self.episode_reward = 0
        self.lane_wait_times = [0, 0, 0, 0]
        self.lane_counts = [0, 0, 0, 0]
        
        return self._get_state()
    
    def _get_state(self):
        """
        Construct state vector for RL agent
        State: [car_counts(4), avg_wait_times(4), light_encoding(4), ambulance_flags(4)]
        Total: 16 dimensions
        """
        car_counts = [0, 0, 0, 0]
        wait_times = [0, 0, 0, 0]
        ambulance_flags = [0, 0, 0, 0]
        
        for v in self.vehicles:
            car_counts[v.lane] += 1
            wait_times[v.lane] += v.wait_time
            if v.is_ambulance:
                ambulance_flags[v.lane] = 1
        
        # Normalize wait times by count
        for i in range(4):
            if car_counts[i] > 0:
                wait_times[i] /= car_counts[i]
        
        # One-hot encode current green light
        light_state = [0, 0, 0, 0]
        light_state[self.current_green] = 1
        
        # Normalize counts (assuming max 20 cars per lane)
        normalized_counts = [min(c / 20.0, 1.0) for c in car_counts]
        normalized_waits = [min(w / 100.0, 1.0) for w in wait_times]
        
        state = normalized_counts + normalized_waits + light_state + ambulance_flags
        return np.array(state, dtype=np.float32)
    
    def _spawn_vehicles(self):
        """Randomly spawn vehicles in lanes with balanced distribution"""
        # Try to spawn in each lane independently for better distribution
        for lane in range(4):
            # Dynamic spawn rate based on current lane occupancy
            current_count = sum(1 for v in self.vehicles if v.lane == lane)
            
            # Reduce spawn rate if lane is too full
            adjusted_rate = SPAWN_RATE
            if current_count > 5:
                adjusted_rate *= 0.5
            elif current_count < 2:
                adjusted_rate *= 1.5  # Boost spawn if lane is empty
            
            if random.random() < adjusted_rate:
                # Higher ambulance spawn rate for demonstration
                is_ambulance = random.random() < AMBULANCE_SPAWN_RATE
                
                new_vehicle = Vehicle(lane, is_ambulance)
                
                # Check if spawn area is clear
                spawn_clear = True
                for v in self.vehicles:
                    if v.lane == lane and v.rect.colliderect(new_vehicle.rect.inflate(60, 60)):
                        spawn_clear = False
                        break
                
                if spawn_clear:
                    self.vehicles.append(new_vehicle)
    
    def _calculate_reward(self, action, prev_ambulance_waiting):
        """
        Calculate reward based on current state
        Improved reward with better incentives
        """
        reward = 0
        
        # Count waiting vehicles and ambulances
        waiting_count = sum(1 for v in self.vehicles if v.waiting)
        ambulance_waiting = any(v.is_ambulance and v.waiting for v in self.vehicles)
        ambulance_moving = any(v.is_ambulance and not v.waiting for v in self.vehicles)
        
        # Base penalty for waiting cars
        reward += REWARDS['per_waiting_car'] * waiting_count
        
        # Switching penalty (encourage stability)
        if action == 1:
            reward += REWARDS['switch_penalty']
        
        # CRITICAL: Ambulance priorities
        if ambulance_moving:
            reward += REWARDS['ambulance_moving']
        if ambulance_waiting:
            reward += REWARDS['ambulance_waiting']
            # Extra penalty if ambulance just started waiting
            if not prev_ambulance_waiting:
                reward -= 75
        
        # Penalty for keeping green light on empty lane
        current_lane_count = sum(1 for v in self.vehicles if v.lane == self.current_green)
        if current_lane_count == 0 and not self.is_yellow:
            reward += REWARDS['empty_lane_penalty']
        
        # Reward for having traffic in current green lane
        if current_lane_count > 0 and action == 0:  # Staying green with traffic
            reward += 5
        
        # Congestion penalty
        for lane in range(4):
            lane_count = sum(1 for v in self.vehicles if v.lane == lane)
            if lane_count > CONGESTION_THRESHOLD:
                reward += REWARDS['congestion_penalty'] * (lane_count - CONGESTION_THRESHOLD)
        
        return reward
    
    def step(self, action):
        """
        Execute one time step
        Returns: (next_state, reward, done, info)
        """
        self.total_steps += 1
        prev_ambulance_waiting = any(v.is_ambulance and v.waiting for v in self.vehicles)
        
        # Check for ambulance in any lane
        ambulance_lane = -1
        for v in self.vehicles:
            if v.is_ambulance:
                ambulance_lane = v.lane
                break
        
        # AMBULANCE PRIORITY: Force immediate switch if ambulance detected
        if AMBULANCE_PRIORITY and ambulance_lane != -1 and ambulance_lane != self.current_green:
            if not self.is_yellow:
                # Calculate how many switches needed to reach ambulance lane
                switches_needed = (ambulance_lane - self.current_green) % 4
                
                # If ambulance is in next lane or we can override min time, switch immediately
                if switches_needed == 1 or AMBULANCE_OVERRIDE_MIN_TIME:
                    self.is_yellow = True
                    self.yellow_timer = 0
                    action = 1  # Override action to switch
        
        # Handle light phases
        if self.is_yellow:
            self.yellow_timer += 1
            if self.yellow_timer >= YELLOW_DURATION:
                # Switch to next lane
                next_lane = (self.current_green + 1) % 4
                
                # If ambulance priority and ambulance is in next lane, go there
                if AMBULANCE_PRIORITY and ambulance_lane != -1:
                    # Calculate shortest path to ambulance
                    direct_dist = (ambulance_lane - self.current_green) % 4
                    if direct_dist <= 2:  # Ambulance is within 2 switches
                        next_lane = (self.current_green + 1) % 4
                
                self.current_green = next_lane
                self.is_yellow = False
                self.yellow_timer = 0
                self.time_in_phase = 0
        else:
            self.time_in_phase += 1
            
            # Force switch if max green time exceeded
            if self.time_in_phase >= MAX_GREEN_TIME:
                self.is_yellow = True
                self.yellow_timer = 0
            # Execute AI action: switch light
            elif action == 1 and self.time_in_phase >= MIN_GREEN_TIME:
                self.is_yellow = True
                self.yellow_timer = 0
            # Allow emergency override even before MIN_GREEN_TIME
            elif action == 1 and AMBULANCE_OVERRIDE_MIN_TIME and ambulance_lane != -1:
                self.is_yellow = True
                self.yellow_timer = 0
        
        # Spawn new vehicles
        self._spawn_vehicles()
        
        # Update all vehicles
        for vehicle in self.vehicles[:]:
            vehicle.update(self.current_green, self.is_yellow, self.vehicles)
            
            # Remove vehicles that exited
            if vehicle.is_off_screen():
                self.vehicles.remove(vehicle)
                self.cars_passed += 1
                if vehicle.is_ambulance:
                    self.ambulances_passed += 1
        
        # Calculate reward
        reward = self._calculate_reward(action, prev_ambulance_waiting)
        self.episode_reward += reward
        
        # Update lane statistics
        self.lane_counts = [0, 0, 0, 0]
        self.lane_wait_times = [0, 0, 0, 0]
        for v in self.vehicles:
            self.lane_counts[v.lane] += 1
            self.lane_wait_times[v.lane] += v.wait_time
        
        # Episode termination
        done = self.total_steps >= MAX_STEPS_PER_EPISODE
        
        info = {
            'total_vehicles': len(self.vehicles),
            'waiting_vehicles': sum(1 for v in self.vehicles if v.waiting),
            'cars_passed': self.cars_passed,
            'ambulances_passed': self.ambulances_passed,
            'avg_wait': np.mean([v.wait_time for v in self.vehicles]) if self.vehicles else 0,
            'has_ambulance': any(v.is_ambulance for v in self.vehicles),
            'ambulance_lane': ambulance_lane
        }
        
        return self._get_state(), reward, done, info
    
    def get_statistics(self):
        """Return current environment statistics"""
        total_cars = len(self.vehicles)
        waiting = sum(1 for v in self.vehicles if v.waiting)
        avg_wait = np.mean([v.total_wait for v in self.vehicles]) if self.vehicles else 0
        
        ambulance_lane = -1
        for v in self.vehicles:
            if v.is_ambulance:
                ambulance_lane = v.lane
                break
        
        return {
            'total_cars': total_cars,
            'waiting_cars': waiting,
            'avg_wait_time': avg_wait,
            'cars_passed': self.cars_passed,
            'ambulances_passed': self.ambulances_passed,
            'current_green': self.current_green,
            'is_yellow': self.is_yellow,
            'lane_counts': self.lane_counts.copy(),
            'has_ambulance': any(v.is_ambulance for v in self.vehicles),
            'ambulance_lane': ambulance_lane
        }