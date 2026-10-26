
import random
import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces


class DinoEnv(gym.Env):
    """Dino game environment for PPO reinforcement learning."""

    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None):
        super().__init__()

        self.render_mode = render_mode
        self.width = 800
        self.height = 400
        self.ground_y = 320

        # Actions: 0 = run, 1 = jump
        self.action_space = spaces.Discrete(2)

        # Observations:
        # distance, obstacle height, jump height,
        # vertical velocity, current speed
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, -20, 6], dtype=np.float32),
            high=np.array([800, 100, 400, 20, 12], dtype=np.float32),
            dtype=np.float32,
        )

        self.screen = None
        self.clock = None
        self.font = None

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.dino_x = 80
        self.dino_size = 45
        self.dino_y = float(self.ground_y - self.dino_size)
        self.dino_velocity = 0.0

        self.obstacle_width = 25
        self.obstacle_x = float(self.width + random.randint(150, 300))
        self.obstacle_height = random.randint(30, 55)

        self.speed = 6.0
        self.score = 0
        self.frame_count = 0

        return self._get_observation(), {}

    def _get_observation(self):
        distance = max(0, self.obstacle_x - self.dino_x)
        height_above_ground = (
            self.ground_y - (self.dino_y + self.dino_size)
        )

        return np.array(
            [
                min(distance, 800),
                self.obstacle_height,
                height_above_ground,
                self.dino_velocity,
                self.speed,
            ],
            dtype=np.float32,
        )

    def step(self, action):
        reward = 0.1
        self.frame_count += 1

        on_ground = (
            self.dino_y >= self.ground_y - self.dino_size
        )

        if int(action) == 1 and on_ground:
            self.dino_velocity = -15.0

        # Apply gravity
        self.dino_velocity += 0.8
        self.dino_y += self.dino_velocity

        if self.dino_y >= self.ground_y - self.dino_size:
            self.dino_y = float(self.ground_y - self.dino_size)
            self.dino_velocity = 0.0

        # Move obstacle
        self.obstacle_x -= self.speed

        # Award points when an obstacle is passed
        if self.obstacle_x + self.obstacle_width < self.dino_x:
            self.score += 1
            reward += 10.0

            # Increase speed gradually, up to 12
            self.speed = min(12.0, self.speed + 0.15)

            # Spawn the next obstacle with a variable gap
            self.obstacle_x = float(
                self.width + random.randint(180, 320)
            )
            self.obstacle_height = random.randint(30, 55)

        # Collision detection
        dino_rect = pygame.Rect(
            self.dino_x + 5,
            int(self.dino_y) + 2,
            35,
            43,
        )

        obstacle_rect = pygame.Rect(
            int(self.obstacle_x),
            self.ground_y - self.obstacle_height,
            self.obstacle_width,
            self.obstacle_height,
        )

        terminated = dino_rect.colliderect(obstacle_rect)

        if terminated:
            reward = -10.0

        # Stop after 3000 steps if no collision occurred
        truncated = self.frame_count >= 3000 and not terminated

        if self.render_mode == "human":
            self.render()

        info = {
            "score": self.score,
            "speed": self.speed,
        }

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            info,
        )

    def _draw_dinosaur(self):
        x = int(self.dino_x)
        y = int(self.dino_y)

        green = (45, 105, 65)
        dark_green = (30, 75, 45)
        white = (255, 255, 255)
        black = (25, 25, 25)

        # Body
        pygame.draw.rect(self.screen, green, (x + 5, y + 17, 27, 22))

        # Neck and head
        pygame.draw.rect(self.screen, green, (x + 22, y + 5, 15, 22))
        pygame.draw.rect(self.screen, green, (x + 22, y + 2, 22, 17))

        # Snout and eye
        pygame.draw.rect(self.screen, green, (x + 35, y + 11, 10, 8))
        pygame.draw.rect(self.screen, white, (x + 35, y + 5, 5, 5))
        pygame.draw.rect(self.screen, black, (x + 37, y + 6, 2, 3))

        # Tail and arms
        pygame.draw.polygon(
            self.screen,
            dark_green,
            [(x + 7, y + 22), (x - 5, y + 14), (x + 5, y + 30)],
        )
        pygame.draw.rect(self.screen, dark_green, (x + 27, y + 24, 10, 4))

        # Legs
        if self.dino_y < self.ground_y - self.dino_size:
            pygame.draw.rect(self.screen, dark_green, (x + 12, y + 36, 7, 7))
            pygame.draw.rect(self.screen, dark_green, (x + 26, y + 36, 7, 7))
        elif (self.frame_count // 6) % 2 == 0:
            pygame.draw.rect(self.screen, dark_green, (x + 10, y + 36, 7, 9))
            pygame.draw.rect(self.screen, dark_green, (x + 26, y + 36, 7, 6))
        else:
            pygame.draw.rect(self.screen, dark_green, (x + 10, y + 36, 7, 6))
            pygame.draw.rect(self.screen, dark_green, (x + 26, y + 36, 7, 9))

    def render(self):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("T-Rex Dino RL - PPO Agent")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 25)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

        self.screen.fill((247, 249, 250))

        pygame.draw.line(
            self.screen, (90, 90, 90),
            (0, self.ground_y), (self.width, self.ground_y), 3
        )

        for mark_x in range(20, self.width, 100):
            pygame.draw.line(
                self.screen, (190, 190, 190),
                (mark_x, self.ground_y + 10),
                (mark_x + 15, self.ground_y + 10), 2
            )

        self._draw_dinosaur()

        # Draw cactus
        cactus_x = int(self.obstacle_x)
        cactus_y = self.ground_y - self.obstacle_height
        cactus_green = (40, 130, 65)

        pygame.draw.rect(
            self.screen, cactus_green,
            (cactus_x + 7, cactus_y, 12, self.obstacle_height)
        )

        if self.obstacle_height >= 35:
            pygame.draw.rect(
                self.screen, cactus_green,
                (cactus_x, cactus_y + 15, 8, 6)
            )
            pygame.draw.rect(
                self.screen, cactus_green,
                (cactus_x, cactus_y + 9, 5, 12)
            )

        score_text = self.font.render(
            f"Score: {self.score}", True, (35, 35, 35)
        )
        speed_text = self.font.render(
            f"Speed: {self.speed:.1f}", True, (35, 35, 35)
        )
        title_text = self.font.render(
            "T-REX REINFORCEMENT LEARNING",
            True, (100, 100, 100)
        )

        self.screen.blit(score_text, (20, 20))
        self.screen.blit(speed_text, (20, 50))
        self.screen.blit(title_text, (350, 20))

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            self.screen = None
            self.clock = None
            self.font = None
