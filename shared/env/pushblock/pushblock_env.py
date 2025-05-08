import numpy as np
import gym
from gym import spaces

class PushBlockEnv(gym.Env):
    """
    A simple environment for the PushBlock task.
    This is a placeholder implementation that returns random observations and rewards.
    """
    
    def __init__(self, seed=0):
        super().__init__()
        
        # Set random seed
        self.np_random = np.random.RandomState(seed)
        
        # Define observation and action spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(7,), dtype=np.float32
        )
        
        # Initialize state
        self.state = None
        self.steps = 0
        self.max_steps = 300
        
    def reset(self):
        """Reset the environment and return the initial observation"""
        self.steps = 0
        # Random initial position
        self.state = self.np_random.uniform(low=-0.5, high=0.5, size=(6,))
        return self.state
        
    def step(self, action):
        """
        Take a step in the environment
        
        Args:
            action: Action to take
            
        Returns:
            observation, reward, done, info
        """
        self.steps += 1
        
        # Simple dynamics: state changes based on action
        self.state[:2] += action[:2] * 0.1  # Use only position components of action
        
        # Add some random noise
        self.state += self.np_random.normal(0, 0.01, size=self.state.shape)
        
        # Calculate reward (distance to target)
        target = np.array([0.3, 0.3, 0.0, 0.0, 0.0, 0.0])
        distance = np.linalg.norm(self.state[:2] - target[:2])
        reward = -distance
        
        # Calculate coverage (negative distance to target, normalized)
        coverage = max(0, 1.0 - distance / 1.0)
        
        # Episode is done if maximum steps reached
        done = self.steps >= self.max_steps
        
        # Information dictionary
        info = {
            'distance': distance,
            'coverage': coverage
        }
        
        return self.state, reward, done, info
        
    def render(self, mode='rgb_array'):
        """
        Render the environment
        
        Args:
            mode: Rendering mode
            
        Returns:
            RGB array if mode is 'rgb_array'
        """
        if mode != 'rgb_array':
            return
            
        # Create a simple 2D visualization
        img_size = 128
        img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        
        # Background color (light gray)
        img.fill(240)
        
        # Convert positions to pixel coordinates
        def pos_to_pixel(pos):
            # Map [-1, 1] to [10, img_size-10]
            x = int((pos[0] + 1) / 2 * (img_size - 20) + 10)
            y = int((pos[1] + 1) / 2 * (img_size - 20) + 10)
            return min(max(x, 0), img_size-1), min(max(y, 0), img_size-1)
            
        # Draw target (red circle)
        target_pos = [0.3, 0.3]
        tx, ty = pos_to_pixel(target_pos)
        cv2_available = False
        try:
            import cv2
            cv2_available = True
            cv2.circle(img, (tx, ty), 10, (0, 0, 255), -1)
        except ImportError:
            # Fallback if cv2 is not available
            for dx in range(-5, 6):
                for dy in range(-5, 6):
                    if dx*dx + dy*dy <= 25:  # Circle radius 5
                        px, py = tx + dx, ty + dy
                        if 0 <= px < img_size and 0 <= py < img_size:
                            img[py, px] = [0, 0, 255]  # Red
            
        # Draw robot (blue circle)
        rx, ry = pos_to_pixel(self.state[:2])
        if cv2_available:
            cv2.circle(img, (rx, ry), 8, (255, 0, 0), -1)
        else:
            for dx in range(-4, 5):
                for dy in range(-4, 5):
                    if dx*dx + dy*dy <= 16:  # Circle radius 4
                        px, py = rx + dx, ry + dy
                        if 0 <= px < img_size and 0 <= py < img_size:
                            img[py, px] = [255, 0, 0]  # Blue
            
        return img 