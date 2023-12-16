import gym
from gym import spaces
import numpy as np
import cv2
import pygame
from setuptools import setup




class MoonLandingEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}

    def __init__(self, render_mode=None):
        super().__init__()
        file_path = 'C:\Python\Autonomous-Landing-Neural-Net\ldem_4.jpg'  # Update this path to where the file is located.
        self.image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        self.elevation_array = np.array(self.image)
        self.size = self.elevation_array.shape

        # Observations are just the agent's location.
        self.observation_space = spaces.Box(low=0, high=max(self.size), shape=(2,), dtype=np.int64)
        # We have 5 actions: 4 movements + 1 terminate
        self.action_space = spaces.Discrete(5)

        self.agent_location = None
        self.render_mode = render_mode
        self.window = None
        self.clock = None

    def reset(self):
        self.agent_location = (np.random.randint(self.size[0]), np.random.randint(self.size[1]))
        return np.array(self.agent_location)

    def step(self, action):
        action_to_direction = {
            0: np.array([0, 1]),    # Move right
            1: np.array([-1, 0]),   # Move up
            2: np.array([0, -1]),   # Move left
            3: np.array([1, 0]),    # Move down
            4: np.array([0, 0])     # Terminate
        }

        terminated = False
        if action == 4:  # Terminate action
            terminated = True
        else:
            # Update the agent's location based on the action
            direction = action_to_direction[action]
            self.agent_location = np.clip(self.agent_location + direction, [0, 0], np.array(self.size) - 1)

        reward = self.calculate_reward(self.elevation_array, self.agent_location)
        observation = np.array(self.agent_location)
        info = {}
        return observation, reward, terminated, info

    def calculate_reward(self, elevation_array, agent_location, radius=1): #We can change radius
        x, y = agent_location
        surroundings = elevation_array[max(x - radius, 0):min(x + radius + 1, self.size[0]),
                                       max(y - radius, 0):min(y + radius + 1, self.size[1])]
        average_surrounding_value = np.mean(surroundings)
        threshold = 100
        deviation = np.sum(np.abs(surroundings - average_surrounding_value))
        reward = max(0, threshold - deviation) / threshold
        return reward

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        
    def close(self):
        if self.window is not None:
            terminated = False
            pygame.display.quit()
            pygame.quit()