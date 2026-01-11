"""
Vehicle Module - Handles individual vehicle behavior
"""
import pygame
from config import SCREEN_WIDTH, SCREEN_HEIGHT, LANE_WIDTH, SAFE_DISTANCE

class Vehicle:
    """Represents a single vehicle in the simulation"""
    
    def __init__(self, lane, is_ambulance=False):
        self.lane = lane  # 0: North, 1: East, 2: South, 3: West
        self.is_ambulance = is_ambulance
        self.speed = 7.5 if is_ambulance else 5.5  # Slightly slower for better visibility
        self.waiting = False
        self.wait_time = 0
        self.total_wait = 0
        
        # Physics
        self.x, self.y = 0, 0
        self.direction = (0, 0)
        self.width = 50
        self.height = 25
        
        self._spawn()
        self.rect = pygame.Rect(0, 0, self.width, self.height)
        self.rect.center = (self.x, self.y)
    
    def _spawn(self):
        """Initialize vehicle position based on lane"""
        cx, cy = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2
        offset = LANE_WIDTH // 4
        
        spawn_configs = {
            0: (cx + offset, -80, (0, 1)),   # North -> South
            1: (SCREEN_WIDTH + 80, cy + offset, (-1, 0)),  # East -> West
            2: (cx - offset, SCREEN_HEIGHT + 80, (0, -1)), # South -> North
            3: (-80, cy - offset, (1, 0))    # West -> East
        }
        
        self.x, self.y, self.direction = spawn_configs[self.lane]
    
    def get_stop_line_position(self):
        """Get the stop line coordinate for this lane"""
        cx, cy = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2
        hw = LANE_WIDTH // 2
        
        stop_lines = {
            0: cy - hw - 15,  # North lane Y
            1: cx + hw + 15,  # East lane X
            2: cy + hw + 15,  # South lane Y
            3: cx - hw - 15   # West lane X
        }
        return stop_lines[self.lane]
    
    def should_stop_at_light(self, current_green, is_yellow):
        """Determine if vehicle should stop based on light"""
        is_my_green = (current_green == self.lane)
        stop_pos = self.get_stop_line_position()
        
        # Red light - always stop before line
        if not is_my_green:
            if self.lane == 0 and self.y < stop_pos:
                return True
            elif self.lane == 1 and self.x > stop_pos:
                return True
            elif self.lane == 2 and self.y > stop_pos:
                return True
            elif self.lane == 3 and self.x < stop_pos:
                return True
        
        # Yellow light - stop if haven't crossed line yet
        elif is_yellow:
            if self.lane == 0 and self.y < stop_pos:
                return True
            elif self.lane == 1 and self.x > stop_pos:
                return True
            elif self.lane == 2 and self.y > stop_pos:
                return True
            elif self.lane == 3 and self.x < stop_pos:
                return True
        
        return False
    
    def check_collision(self, vehicles):
        """Check if there's a vehicle ahead requiring us to stop"""
        for other in vehicles:
            if other == self or other.lane != self.lane:
                continue
            
            # Calculate distance to vehicle ahead
            dist = float('inf')
            if self.lane == 0 and other.y > self.y:
                dist = other.y - self.y
            elif self.lane == 1 and other.x < self.x:
                dist = self.x - other.x
            elif self.lane == 2 and other.y < self.y:
                dist = self.y - other.y
            elif self.lane == 3 and other.x > self.x:
                dist = other.x - self.x
            
            if 0 < dist < SAFE_DISTANCE:
                return True
        
        return False
    
    def update(self, current_green, is_yellow, vehicles):
        """Update vehicle position and state"""
        must_stop = False
        
        # Check traffic light
        if self.should_stop_at_light(current_green, is_yellow):
            must_stop = True
        
        # Check collision with other vehicles
        if self.check_collision(vehicles):
            must_stop = True
        
        # Update position
        if must_stop:
            self.waiting = True
            self.wait_time += 1
            self.total_wait += 1
        else:
            self.waiting = False
            self.wait_time = 0
            self.x += self.direction[0] * self.speed
            self.y += self.direction[1] * self.speed
        
        self.rect.center = (self.x, self.y)
    
    def is_off_screen(self):
        """Check if vehicle has left the simulation area"""
        margin = 150
        return not (-margin <= self.x <= SCREEN_WIDTH + margin and 
                   -margin <= self.y <= SCREEN_HEIGHT + margin)
    
    def draw(self, screen, car_images, amb_images):
        """Draw the vehicle on screen"""
        image_dict = amb_images if self.is_ambulance else car_images
        image = image_dict[self.lane]
        rect = image.get_rect(center=(self.x, self.y))
        screen.blit(image, rect)