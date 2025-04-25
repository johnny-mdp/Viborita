import pygame
import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
import pickle
from typing import List, Optional
import imageio


screenpx: int = 600
grid: int = 20
cell: int = screenpx // grid
batch_num: int = 128
epochs: int = 100
run_max: int = 1000
path_save: str = "best_model.pth"


#movements the snake can make
left: np.ndarray = np.array([-1, 0])
right: np.ndarray = np.array([1, 0])
up: np.ndarray = np.array([0, -1])
low: np.ndarray = np.array([0, 1])
dirs: List[np.ndarray] = [left, right, up, low]



class NeuralNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        """
        Ad hoc structure. I found this produces acceptable results.
        """
        self.net: nn.Sequential = nn.Sequential(
            nn.Linear(7, 12),
            nn.ReLU(),
            nn.Linear(12, 8),
            nn.ReLU(),
            nn.Linear(8, 3)
        )

        for layer in self.net:
            """
            Applies weight decay (results in faster learning)
            """
            if isinstance(layer, nn.Linear):
                fan_in: int = layer.in_features
                std: float = 1 / math.sqrt(fan_in)
                torch.nn.init.normal_(layer.weight, mean=0.0, std=std)
                torch.nn.init.normal_(layer.bias, mean=0.0, std=std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)




class Snake:
    def __init__(self) -> None:
        """
        A snake:
        - has a body
        - moves in a certain direction
        - is alive or dead
        - a list that chacks for possible loops
        """
        self.body: List[np.ndarray] = [np.array([grid // 2, grid // 2])]
        self.direction: np.ndarray = random.choice(dirs)
        self.alive: bool = True
        self.history: List[List[int]] = []

    def move(self) -> None:
        """
        Moves snake
        """
        new_head: np.ndarray = self.body[0] + self.direction
        self.body.insert(0, new_head)
        self.body.pop()
        self.history.append(self.direction.tolist())
        if len(self.history) > 4:
            self.history.pop(0)

        if not (0 <= new_head[0] < grid and 0 <= new_head[1] < grid):
            self.alive = False

    def grow(self) -> None:
        """
        Updates snake's lenght
        """
        self.body.append(self.body[-1])

    def is_repeating(self) -> bool:
        """
        Discovers if the snake is moving in a loop
        """
        return len(self.history) == 4 and all(
            np.array_equal(d, self.history[0]) for d in self.history
        )

class Game:
    def __init__(self, player: Optional[NeuralNet] = None) -> None:
        """
        A game consists of:
        - a snake
        - an apple
        - a player (the NN)
        - a score
        - a num of steps (so it does 
          not play indefinetely)
        """
        self.snake: Snake = Snake()
        self.apple: np.ndarray = self.spawn_apple()
        self.player: NeuralNet = player or NeuralNet()
        self.score: int = 0
        self.steps: int = 0
        self.last_distance: float = self.distance_to_apple()

    def distance_to_apple(self) -> float:
        """
        Calcualtes distance from snake's head
        to apple with an L2 metric
        """
        return float(np.linalg.norm(self.snake.body[0] - self.apple))

    def spawn_apple(self) -> np.ndarray:
        """
        Creates an apple in a random position
        """
        return np.random.randint(0, grid, size=2)

    def step(self, train: bool = False) -> int:
        """
        Snake decides movement, moves and 
        recieves scores/punishments.
        """
        input_tensor: torch.Tensor = self.get_inputs()
        output: torch.Tensor = self.player(input_tensor)

        decision: int = torch.argmax(output).item()
        self.update_direction(decision)
        self.snake.move()
        self.steps += 1

        if not self.snake.alive:
            return -100

        score: int = 0
        if np.array_equal(self.snake.body[0], self.apple):
            score += 100
            self.snake.grow()
            self.apple = self.spawn_apple()
            self.last_distance = self.distance_to_apple()
        else:
            new_distance: float = self.distance_to_apple()
            if new_distance < self.last_distance:
                score += 5
            elif new_distance == self.last_distance:
                score += 1
            else:
                score -= 5
            self.last_distance = new_distance

        if self.snake.is_repeating():
            score -= 50

        return score

    def get_inputs(self) -> torch.Tensor:
        """
        Gets snake's, position, speed, apple's position
        and the distance between them
        """
        head: np.ndarray = self.snake.body[0]
        left = head + np.array([-1, 0])
        right = head + np.array([1, 0])
        up = head + np.array([0, -1])
        low = head + np.array([0, 1])
        inputs: List[float] = [
            float(int((left < 0).any() or (left >= grid).any())),
            float(int((right < 0).any() or (right >= grid).any())),
            float(int((up < 0).any() or (up >= grid).any())),
            float(self.snake.direction[0]),
            float(self.snake.direction[1]),
            float(self.apple[0] - head[0]),
            float(self.apple[1] - head[1])
        ]
        return torch.tensor(inputs, dtype=torch.float32)

    def update_direction(self, decision: int) -> None:
        """
        Applies snake's new movement and changes its direction
        """
        if decision == 0:
            self.snake.direction = np.array([-self.snake.direction[1], self.snake.direction[0]])
        elif decision == 2:
            self.snake.direction = np.array([self.snake.direction[1], -self.snake.direction[0]])


def train_model(adapt: bool = True) -> NeuralNet:
    """
    Trains model.
    adapt= True: learning rate decay 
    """
    model: NeuralNet = NeuralNet()
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    criterion = nn.MSELoss()

    trained: bool = False
    for epoch in range(epochs):
        game: Game = Game(model)
        total_score: int = 0
        if epoch > 0 and epoch % 50 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] /= 2
        for _ in range(run_max):
            state: torch.Tensor = game.get_inputs()
            q_values: torch.Tensor = model(state)
            action: int = torch.argmax(q_values).item()
            score: int = game.step()

            next_state: torch.Tensor = game.get_inputs()
            next_q_values: torch.Tensor = model(next_state)
            max_next_q: torch.Tensor = torch.max(next_q_values).detach()
            target_q: torch.Tensor = q_values.clone()
            target_q[action] = score + 0.9 * max_next_q

            loss = criterion(q_values, target_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_score += score
            if not game.snake.alive:
                break
            if total_score > 3000:
                #Ad hoc value assumed to be good enough
                trained = True
                break
        print(f"Epoch {epoch+1}, Total score: {total_score}")
        if trained:
            break

    return model


def run_game(model: NeuralNet, save_gif: bool = False, gif_path: str = 'snake_game.gif') -> None:
    """
    Runs game with the trained model of the last epoch.
    Save GIF of the trained NN playing.
    """
    pygame.init()
    screen = pygame.display.set_mode((screenpx, screenpx))
    clock = pygame.time.Clock()
    game: Game = Game(model)

    frames: List[np.ndarray] = [] if save_gif else []
    running: bool = True
    while running and game.snake.alive and game.steps < run_max:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill((0, 0, 0))
        pygame.draw.rect(screen, (255, 0, 0), (*game.apple * cell, cell, cell))
        for part in game.snake.body:
            pygame.draw.rect(screen, (0, 255, 0), (*part * cell, cell, cell))

        pygame.display.flip()

        if save_gif:
            frame = pygame.surfarray.array3d(screen)
            frames.append(np.transpose(frame, (1, 0, 2)))

        game.step()
        clock.tick(10)

    pygame.quit()
    if save_gif:
        imageio.mimsave(gif_path, frames, fps=10)


if __name__ == "__main__":
    trained_model: NeuralNet = train_model()

    sv: str = input("Save model? y/n \n")
    i: int = 0
    while (sv.lower() != 'y' and sv.lower() != 'n') and i < 10:
        sv = input("Invalid input. Try again.\n")
        i += 1
    if i == 10:
        print("Couldn't save model. :(")
    if sv.lower() == 'y':
        torch.save(trained_model.state_dict(), path_save)

    run_game(trained_model, True, 'snake_gameplay.gif')
