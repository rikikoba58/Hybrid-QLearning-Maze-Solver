import pygame
import random
import numpy as np

# --- 設定 ---
SCREEN_WIDTH = 500
SCREEN_HEIGHT = 780 
GRID_SIZE = 10
CELL_SIZE = SCREEN_WIDTH // GRID_SIZE

# 色の定義
WHITE, BLACK, BLUE, GREEN, RED = (255, 255, 255), (0, 0, 0), (0, 0, 255), (0, 255, 0), (255, 50, 50)
TEXT_COLOR, Q_TEXT_COLOR = (30, 30, 30), (50, 50, 50)

def generate_maze(size, wall_prob=0.2):
    maze = [[0 for _ in range(size)] for _ in range(size)]
    for y in range(size):
        for x in range(size):
            if (y == 0 and x == 0) or (y == size-1 and x == size-1): continue
            if random.random() < wall_prob: maze[y][x] = 1
    maze[size-1][size-1] = 2
    return maze

class HybridAgent:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Q-Learning x Genetic Algorithm Hybrid")
        self.font = pygame.font.SysFont("Meiryo", 16)
        self.small_font = pygame.font.SysFont("Arial", 11)
        
        self.actions = [[0, 1], [0, -1], [1, 0], [-1, 0], [0, 0]]
        self.alpha, self.gamma, self.epsilon = 0.5, 0.9, 0.1
        
        # すべてのパラメータを初期化
        self.full_reset()

    def full_reset(self):
        """迷路、学習データ、GAパラメータすべてをリセット"""
        self.maze = generate_maze(GRID_SIZE)
        self.q_table = np.zeros((GRID_SIZE, GRID_SIZE, 5))
        self.obs_pos = [random.randint(3, 6), random.randint(2, 7), 1]
        
        self.generation = 1
        self.mutation_rate = 0.05
        self.history = []
        self.total_episodes = 0
        self.conflicts = 0
        self.avg_steps = 0.0
        self.last_result = "Reset All"
        
        self.reset_agent_position()

    def reset_agent_position(self):
        """エージェントの位置のみをリセット（エピソード毎用）"""
        self.pos = [0, 0]
        self.done = False
        self.step_count = 0
        self.total_episodes += 1

    def select_action(self):
        # 遺伝的アルゴリズム的な突然変異
        if random.random() < self.mutation_rate:
            return random.randint(0, 4)
        # Q学習的な探索
        if random.random() < self.epsilon:
            return random.randint(0, 4)
        return np.argmax(self.q_table[self.pos[0], self.pos[1]])

    def update(self):
        if self.done: return None
        self.step_count += 1
        
        if self.step_count % 2 == 0:
            self.obs_pos[1] += self.obs_pos[2]
            if self.obs_pos[1] <= 1 or self.obs_pos[1] >= GRID_SIZE-2: self.obs_pos[2] *= -1

        state = self.pos[:]
        action_idx = self.select_action()
        move = self.actions[action_idx]
        ny, nx = max(0, min(GRID_SIZE-1, state[0]+move[0])), max(0, min(GRID_SIZE-1, state[1]+move[1]))

        reward = -1
        collision_type = None

        if action_idx == 4: # Stayのアクション評価
            dist = abs(self.pos[0]-self.obs_pos[0]) + abs(self.pos[1]-self.obs_pos[1])
            reward = 2 if dist <= 1 else -3

        if self.maze[ny][nx] == 1:
            ny, nx, reward = state[0], state[1], -10
        elif ny == self.obs_pos[0] and nx == self.obs_pos[1]:
            reward, self.done, collision_type = -100, True, "Obstacle"
        elif self.maze[ny][nx] == 2:
            reward, self.done, collision_type = 500, True, "Goal"

        max_f_q = np.max(self.q_table[ny, nx])
        curr_q = self.q_table[state[0], state[1], action_idx]
        self.q_table[state[0], state[1], action_idx] = curr_q + self.alpha * (reward + self.gamma * max_f_q - curr_q)
        
        self.pos = [ny, nx]
        return collision_type

    def draw(self):
        self.screen.fill(WHITE)
        max_q = np.max(self.q_table) if np.max(self.q_table) > 0 else 1
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                v = np.max(self.q_table[y, x])
                rect = (x*CELL_SIZE, y*CELL_SIZE, CELL_SIZE, CELL_SIZE)
                if v > 0:
                    c = min(255, int((v / max_q) * 200))
                    pygame.draw.rect(self.screen, (255, 255-c, 255-c), rect)
                if self.maze[y][x] == 1: pygame.draw.rect(self.screen, BLACK, rect)
                elif self.maze[y][x] == 2: pygame.draw.rect(self.screen, GREEN, rect)
                pygame.draw.rect(self.screen, (220, 220, 220), rect, 1)
                if self.maze[y][x] != 1 and v != 0:
                    q_text = self.small_font.render(f"{v:.1f}", True, Q_TEXT_COLOR)
                    self.screen.blit(q_text, (x*CELL_SIZE + 4, y*CELL_SIZE + 4))
        pygame.draw.circle(self.screen, RED, (self.obs_pos[1]*CELL_SIZE+25, self.obs_pos[0]*CELL_SIZE+25), 15)
        pygame.draw.rect(self.screen, BLUE, (self.pos[1]*CELL_SIZE+12, self.pos[0]*CELL_SIZE+12, 26, 26))

        pygame.draw.rect(self.screen, (240, 240, 240), (0, 500, 500, 280))
        c_rate = (self.conflicts / self.total_episodes) * 100 if self.total_episodes > 0 else 0
        info = [
            f"Generation: {self.generation} | Episode: {self.total_episodes}",
            f"Last Result: {self.last_result}",
            f"Average Steps: {self.avg_steps:.2f} | Conflict: {c_rate:.1f}%",
            f"Evolution: Mutation Rate {self.mutation_rate*100}%",
            "Hybrid Logic: Q-Learning x Genetic Diversity.",
            "[R]: RESET ALL (Maze/Q-Table/GA)  [Space]: Speed Up"
        ]
        for i, text in enumerate(info):
            color = (200, 0, 0) if i == 5 else TEXT_COLOR
            self.screen.blit(self.font.render(text, True, color), (20, 510 + (i * 28)))
        pygame.display.flip()

if __name__ == "__main__":
    game = HybridAgent()
    clock, running, fps = pygame.time.Clock(), True, 60
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: 
                    game.full_reset() # Rキーですべてをリセット
                if event.key == pygame.K_SPACE: 
                    fps = 500 if fps == 60 else 60

        res = game.update(); game.draw()
        if game.done:
            if res == "Obstacle": 
                game.conflicts += 1
                game.last_result = "EXTINCT (Collision)"
            else:
                game.history.append(game.step_count)
                game.avg_steps = sum(game.history) / len(game.history)
                game.last_result = f"SURVIVED ({game.step_count} steps)"
            
            if game.total_episodes % 50 == 0:
                game.generation += 1
                game.mutation_rate = max(0.01, game.mutation_rate * 0.9)
            
            pygame.time.delay(100); game.reset_agent_position()
        clock.tick(fps)
    pygame.quit()