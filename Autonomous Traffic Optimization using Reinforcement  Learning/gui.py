"""
GUI Module - Handles all visualization
"""
import pygame
import os
from config import *

class TrafficGUI:
    """Handles all rendering and visualization"""
    
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("AI Traffic Control System - Deep Q-Learning")
        self.clock = pygame.time.Clock()
        
        # Fonts
        self.title_font = pygame.font.SysFont("Arial", 32, bold=True)
        self.header_font = pygame.font.SysFont("Arial", 24, bold=True)
        self.font = pygame.font.SysFont("Arial", 18)
        self.small_font = pygame.font.SysFont("Arial", 14)
        
        # Load vehicle images
        self.car_images = {}
        self.amb_images = {}
        self._load_assets()
        
        # Animation
        self.frame_count = 0
    
    def _load_assets(self):
        """Load or create vehicle sprites"""
        def create_car_sprite(color, size=(50, 25)):
            surf = pygame.Surface(size, pygame.SRCALPHA)
            # Body
            pygame.draw.rect(surf, color, (5, 5, size[0]-10, size[1]-10), border_radius=8)
            # Windows
            window_color = (100, 150, 200, 180)
            pygame.draw.rect(surf, window_color, (15, 8, 15, 10), border_radius=3)
            pygame.draw.rect(surf, window_color, (size[0]-25, 8, 15, 10), border_radius=3)
            # Lights
            pygame.draw.circle(surf, (255, 255, 200), (size[0]-8, size[1]//2), 3)
            return surf
        
        # Create base sprites
        base_car = create_car_sprite(COLORS['car'])
        base_amb = create_car_sprite(COLORS['ambulance'])
        
        # Add cross to ambulance
        cross_surf = pygame.Surface((50, 25), pygame.SRCALPHA)
        pygame.draw.line(cross_surf, (255, 255, 255), (20, 12), (30, 12), 3)
        pygame.draw.line(cross_surf, (255, 255, 255), (25, 7), (25, 17), 3)
        base_amb.blit(cross_surf, (0, 0))
        
        # Rotate for each direction
        rotations = {0: -90, 1: 180, 2: 90, 3: 0}
        for lane, angle in rotations.items():
            self.car_images[lane] = pygame.transform.rotate(base_car, angle)
            self.amb_images[lane] = pygame.transform.rotate(base_amb, angle)
    
    def handle_events(self):
        """Process pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
        return True
    
    def draw_intersection(self):
        """Draw road intersection"""
        cx, cy = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2
        hw = LANE_WIDTH // 2
        
        # Roads
        pygame.draw.rect(self.screen, COLORS['road'], 
                        (cx - hw, 0, LANE_WIDTH, SCREEN_HEIGHT))
        pygame.draw.rect(self.screen, COLORS['road'], 
                        (0, cy - hw, SCREEN_WIDTH, LANE_WIDTH))
        
        # Lane dividers
        pygame.draw.line(self.screen, COLORS['line'], 
                        (cx, 0), (cx, cy - hw), 3)
        pygame.draw.line(self.screen, COLORS['line'], 
                        (cx, cy + hw), (cx, SCREEN_HEIGHT), 3)
        pygame.draw.line(self.screen, COLORS['line'], 
                        (0, cy), (cx - hw, cy), 3)
        pygame.draw.line(self.screen, COLORS['line'], 
                        (cx + hw, cy), (SCREEN_WIDTH, cy), 3)
        
        # Stop lines (dashed)
        dash_length = 15
        for i in range(0, LANE_WIDTH, dash_length * 2):
            # North stop line
            pygame.draw.line(self.screen, COLORS['line'],
                           (cx - hw + i, cy - hw - 10),
                           (cx - hw + i + dash_length, cy - hw - 10), 4)
            # East stop line
            pygame.draw.line(self.screen, COLORS['line'],
                           (cx + hw + 10, cy - hw + i),
                           (cx + hw + 10, cy - hw + i + dash_length), 4)
            # South stop line
            pygame.draw.line(self.screen, COLORS['line'],
                           (cx - hw + i, cy + hw + 10),
                           (cx - hw + i + dash_length, cy + hw + 10), 4)
            # West stop line
            pygame.draw.line(self.screen, COLORS['line'],
                           (cx - hw - 10, cy - hw + i),
                           (cx - hw - 10, cy - hw + i + dash_length), 4)
    
    def draw_traffic_lights(self, current_green, is_yellow):
        """Draw traffic light signals"""
        cx, cy = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2
        hw = LANE_WIDTH // 2
        
        # Light positions
        positions = [
            (cx + hw + 35, cy - hw - 35),  # North
            (cx + hw + 35, cy + hw + 35),  # East
            (cx - hw - 35, cy + hw + 35),  # South
            (cx - hw - 35, cy - hw - 35)   # West
        ]
        
        for i, pos in enumerate(positions):
            # Determine color
            if current_green == i:
                color = COLORS['yellow'] if is_yellow else COLORS['green']
            else:
                color = COLORS['red']
            
            # Draw glow
            glow_radius = 22
            glow_surf = pygame.Surface((glow_radius*2, glow_radius*2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, (*color, 80), (glow_radius, glow_radius), glow_radius)
            self.screen.blit(glow_surf, (pos[0]-glow_radius, pos[1]-glow_radius))
            
            # Draw light
            pygame.draw.circle(self.screen, color, pos, 18)
            pygame.draw.circle(self.screen, COLORS['line'], pos, 18, 3)
    
    def draw_vehicles(self, vehicles):
        """Draw all vehicles"""
        for vehicle in vehicles:
            vehicle.draw(self.screen, self.car_images, self.amb_images)
    
    def draw_dashboard(self, stats, episode, epsilon, loss):
        """Draw information dashboard"""
        dash_x = SCREEN_WIDTH
        
        # Background
        pygame.draw.rect(self.screen, COLORS['dash_bg'], 
                        (dash_x, 0, DASHBOARD_WIDTH, SCREEN_HEIGHT))
        pygame.draw.line(self.screen, COLORS['cyan'], 
                        (dash_x, 0), (dash_x, SCREEN_HEIGHT), 3)
        
        y = 30
        
        # Title
        title = self.title_font.render("TRAFFIC AI", True, COLORS['cyan'])
        self.screen.blit(title, (dash_x + 80, y))
        y += 60
        
        # Lane Status
        header = self.header_font.render("Traffic Lights", True, COLORS['text'])
        self.screen.blit(header, (dash_x + 20, y))
        y += 40
        
        lane_names = ["North", "East ", "South", "West "]
        for i in range(4):
            status = "RED"
            color = COLORS['red']
            if stats['current_green'] == i:
                if stats['is_yellow']:
                    status = "YELLOW"
                    color = COLORS['yellow']
                else:
                    status = "GREEN"
                    color = COLORS['green']
            
            # Lane name
            name_text = self.font.render(f"{lane_names[i]}:", True, COLORS['text'])
            self.screen.blit(name_text, (dash_x + 30, y))
            
            # Status badge
            badge_x = dash_x + 150
            pygame.draw.rect(self.screen, color, 
                           (badge_x, y, 100, 25), border_radius=5)
            status_text = self.font.render(status, True, (0, 0, 0))
            text_rect = status_text.get_rect(center=(badge_x + 50, y + 12))
            self.screen.blit(status_text, text_rect)
            
            # Car count
            count_text = self.small_font.render(
                f"({stats['lane_counts'][i]} cars)", True, COLORS['text'])
            self.screen.blit(count_text, (dash_x + 260, y + 5))
            
            y += 35
        
        y += 30
        
        # Training Stats
        header = self.header_font.render("Training Stats", True, COLORS['text'])
        self.screen.blit(header, (dash_x + 20, y))
        y += 40
        
        info_lines = [
            f"Episode: {episode}",
            f"Total Cars: {stats['total_cars']}",
            f"Waiting: {stats['waiting_cars']}",
            f"Avg Wait: {stats['avg_wait_time']:.1f}",
            f"Cars Passed: {stats['cars_passed']}",
            f"Epsilon: {epsilon:.3f}",
        ]
        
        if loss is not None:
            info_lines.append(f"Loss: {loss:.4f}")
        
        for line in info_lines:
            text = self.font.render(line, True, COLORS['text'])
            self.screen.blit(text, (dash_x + 30, y))
            y += 30
        
        # Ambulance Alert
        if stats['has_ambulance']:
            y += 20
            if self.frame_count % 30 < 15:  # Blink effect
                alert_box = pygame.Rect(dash_x + 20, y, DASHBOARD_WIDTH - 40, 60)
                pygame.draw.rect(self.screen, COLORS['ambulance'], alert_box, border_radius=10)
                pygame.draw.rect(self.screen, COLORS['line'], alert_box, 3, border_radius=10)
                
                alert_text = self.header_font.render("🚨 AMBULANCE", True, COLORS['text'])
                text_rect = alert_text.get_rect(center=(dash_x + DASHBOARD_WIDTH//2, y + 30))
                self.screen.blit(alert_text, text_rect)
    
    def render(self, environment, episode, epsilon, loss=None):
        """Main render function"""
        self.frame_count += 1
        
        # Background
        self.screen.fill(COLORS['background'])
        
        # Draw components
        self.draw_intersection()
        self.draw_traffic_lights(environment.current_green, environment.is_yellow)
        self.draw_vehicles(environment.vehicles)
        
        # Dashboard
        stats = environment.get_statistics()
        self.draw_dashboard(stats, episode, epsilon, loss)
        
        # Update display
        pygame.display.flip()
        self.clock.tick(FPS)
    
    def close(self):
        """Cleanup"""
        pygame.quit()