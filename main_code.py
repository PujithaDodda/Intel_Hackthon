import tkinter as tk
from concurrent.futures import ThreadPoolExecutor
import heapq
import time
from tkinter import messagebox, ttk
from PIL import Image, ImageTk
import torch
import torch.nn as nn
import numpy as np
import math
from openvino.runtime import Core
import random
from tkinter import simpledialog, messagebox

class OpenVINOMazeTransformer:
    def __init__(self, model_path="optimized_model/maze_model.xml"):
        """
        Load the optimized OpenVINO model.
        """
        self.core = Core()
        self.model = self.core.read_model(model=model_path)
        self.compiled_model = self.core.compile_model(model=self.model, device_name="CPU")
        self.input_layer = self.compiled_model.input(0)
        self.output_layer = self.compiled_model.output(0)

    def infer(self, maze_state):
        """
        Perform inference using the OpenVINO model.
        """
        # Convert PyTorch tensor to NumPy array if needed
        maze_state = maze_state.cpu().numpy() if isinstance(maze_state, torch.Tensor) else maze_state
        # Perform inference
        result = self.compiled_model([maze_state])
        # Convert result back to PyTorch tensor for compatibility with game logic
        return torch.tensor(result[self.output_layer], dtype=torch.float32)



# Define Transformer model
class MazeTransformer(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_layers=2):
        super(MazeTransformer, self).__init__()
        self.embedding = nn.Linear(2, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
            nn.Softmax(dim=-1)
        )

    def forward(self, maze_state):
        # Ensure maze_state has correct shape (batch, sequence, features)
        x = self.embedding(maze_state)
        
        # Remove the unnecessary unsqueeze here to keep it as a 3D tensor
        transformer_out = self.transformer(x)
        
        last_hidden = transformer_out[:, -1, :]  # Take last hidden state
        return self.output_layer(last_hidden)

class MazeTransformerWithLSTM(MazeTransformer):
    def __init__(self, d_model=64, nhead=4, num_layers=2):  # Corrected with double underscores
        super(MazeTransformerWithLSTM, self).__init__(d_model, nhead, num_layers)
        self.lstm = nn.LSTM(d_model, d_model, num_layers=1, batch_first=True)

    def forward(self, maze_state):
        x = self.embedding(maze_state)  # Inherits embedding from MazeTransformer
        lstm_out, _ = self.lstm(x)  # Pass x directly to LSTM without unsqueeze(0)
        transformer_out = self.transformer(lstm_out)
        last_hidden = transformer_out[:, -1, :]  # Take last hidden state
        return self.output_layer(last_hidden)
# Maze and game classes
class Maze:
    def __init__(self, rows=15, cols=15, sparsity=0.3):
        self.rows = rows
        self.cols = cols
        self.sparsity = sparsity
        self.grid = self._generate_maze_with_multiple_paths()
        self.agent_row, self.agent_col = 1, 1  # Jerry's initial position
        self.goal = (rows - 2, cols - 2)       # Tom's initial position
        self.grid[self.agent_row][self.agent_col] = 'P'
        self.grid[self.goal[0]][self.goal[1]] = 'E'
        
        self.puzzle_positions = [(5, 5), (10, 10), (15, 15)]  # Locations of the puzzles
        self.cheese_positions = []
        self.hiding_spots = []
        self.power_ups = []  # Power-ups locations
        self.history = [(self.agent_row, self.agent_col)]

    def _generate_maze_with_multiple_paths(self):
        grid = [['#' for _ in range(self.cols)] for _ in range(self.rows)]

        def create_path(row, col):
            grid[row][col] = ' '
            directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]
            random.shuffle(directions)
            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc
                if 0 < new_row < self.rows - 1 and 0 < new_col < self.cols - 1 and grid[new_row][new_col] == '#':
                    grid[row + dr // 2][col + dc // 2] = ' '
                    create_path(new_row, new_col)

        create_path(1, 1)

        # Parallelize wall removal using ThreadPoolExecutor for better performance
        walls = []
        for i in range(1, self.rows - 1):
            for j in range(1, self.cols - 1):
                if grid[i][j] == '#':
                    neighbors = sum(1 for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]
                                    if 0 <= i + di < self.rows and 0 <= j + dj < self.cols and grid[i + di][j + dj] == ' ')
                    if 1 <= neighbors <= 2:
                        walls.append((i, j))

        sparsity=0.05 # Increased wall density (less empty space)

        num_walls_to_remove = int(len(walls) * (1 - self.sparsity))  

        # Use ThreadPoolExecutor to remove walls in parallel
        def remove_wall(wall_index):
            i, j = walls[wall_index]
            grid[i][j] = ' '

        with ThreadPoolExecutor() as executor:
            executor.map(remove_wall, range(num_walls_to_remove))

        return grid
    
    def is_valid_move(self, row, col):
        return 0 <= row < self.rows and 0 <= col < self.cols and self.grid[row][col] in  [' ', 'E', 'C', 'H', 'P']

    def move(self, drow, dcol):
        new_row, new_col = self.agent_row + drow, self.agent_col + dcol
        if self.is_valid_move(new_row, new_col):
            self.grid[self.agent_row][self.agent_col] = ' '
            self.agent_row, self.agent_col = new_row, new_col
            self.grid[self.agent_row][self.agent_col] = 'P'
            self.history.append((new_row, new_col))
            return True
        return False

    def place_cheese(self, num_cheese=5):
        for _ in range(num_cheese):
            while True:
                row, col = random.randint(1, self.rows - 2), random.randint(1, self.cols - 2)
                if self.grid[row][col] == ' ':
                    self.grid[row][col] = 'C'  # Cheese
                    self.cheese_positions.append((row, col))
                    break

    def place_hiding_spots(self, num_spots=3):
        for _ in range(num_spots):
            while True:
                row, col = random.randint(1, self.rows - 2), random.randint(1, self.cols - 2)
                if self.grid[row][col] == ' ':
                    self.grid[row][col] = 'H'  # Hiding spot
                    self.hiding_spots.append((row, col))
                    break

    def place_power_ups(self, num_power_ups=3):
        for _ in range(num_power_ups):
            while True:
                row, col = random.randint(1, self.rows - 2), random.randint(1, self.cols - 2)
                if self.grid[row][col] == ' ':
                    self.grid[row][col] = 'P'  # Power-up
                    self.power_ups.append((row, col))
                    break




class EnhancedMazeGame:
    def __init__(self, rows=15, cols=15, time_limit=60, use_lstm=False,use_openvino=True, openvino_model_path="optimized_model/maze_model.xml"):
        self.rows = rows
        self.cols = cols
        self.root = tk.Tk()
        self.CELL_SIZE = 40
        self.cheese_collected = 0
        self.tom_score = 0  # Initialize Tom's score
        self.cheese_target = 5
        self.jerry_steps = 0  # Counter for Jerry's steps
        self.puzzle_active = False  # Flag to track if puzzle is active
        self.puzzle_answered = False  # Flag to track if the question has been answered
        self.tom_move_timer = 600  # Timer for Tom's movement
        self.tom_move_callback_id = None
        self.jerry_move_timer = 500  # Set a default time for Jerry’s movement delay
        self.jerry_move_callback_id = None  # Store callback ID for Tom's move

        self.level = 1
        self.time_limit = time_limit
        self.game_over = False
        self.speed_up_threshold = 3
        self.base_escape_probability = 100  # Start with 100% escape probability

        self.tom_path = []
        self.use_lstm = use_lstm  # Store use_lstm as an instance variable
        self.current_level_cheese_collected = 0
        self.DIRECTION_LIST = ['Up', 'Right', 'Down', 'Left']
        self.DIRECTIONS = {
            'Up': {'index': 0, 'delta': (-1, 0)},
            'Right': {'index': 1, 'delta': (0, 1)},
            'Down': {'index': 2, 'delta': (1, 0)},
            'Left': {'index': 3, 'delta': (0, -1)}
        }

        # Initialize Tom's movement timer
        self.base_tom_move_timer = 600
        self.tom_move_timer = self.base_tom_move_timer
        self.optimal_path = []

        # Flag to track puzzle state
        self.puzzle_active = False  # Flag to track whether a puzzle is active

        # Load images
        self.jerry_image_path = "C:\\Users\\dodda\\Jerry.png" #"https://drive.google.com/file/d/1RsLEWi0qAuY5KA15iDb4INxisnmz14V1/view?usp=drive_link" this is link for the jerry picture that i have used"
        self.tom_image_path = "C:\\Users\\dodda\\Tom.png" #"https://drive.google.com/file/d/19fdMm253xxXpP2og9I4Y0lAVxXTf-cNl/view?usp=drive_link" tom picture link"
        self.cheese_image_path = "C:\\Users\\dodda\\Cheese.jpg"
        self.jerry_image = ImageTk.PhotoImage(Image.open(self.jerry_image_path).resize((self.CELL_SIZE, self.CELL_SIZE)))
        self.tom_image = ImageTk.PhotoImage(Image.open(self.tom_image_path).resize((self.CELL_SIZE, self.CELL_SIZE)))
        self.cheese_image = ImageTk.PhotoImage(Image.open(self.cheese_image_path).resize((self.CELL_SIZE, self.CELL_SIZE)))
        self.colors = {
            'wall': 'gray',
            'path': 'white',
            'optimal_path': 'blue'
        }

        # Device setup for torch (CPU or GPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize the model based on game mode
        if use_lstm:
            self.model = MazeTransformerWithLSTM().to(self.device)
        else:
            self.model = MazeTransformer().to(self.device)

        # Load a smaller version of Jerry's image for the legend
        self.jerry_image_legend = ImageTk.PhotoImage(Image.open(self.jerry_image_path).resize((20, 20)))

        # Legend items
        self.legend_items = [
            ("Player", self.jerry_image_legend),
            ("Wall", self.colors['wall']),
            ("Path", self.colors['path']),
            ("Optimal Path", self.colors['optimal_path'])
        ]

        # Initialize level and UI
        self._initialize_level()
        self._setup_ui()

        
    def pause_tom(self):
        """Pause Tom's movement while the puzzle is active."""
        if self.tom_move_callback_id:
            self.root.after_cancel(self.tom_move_callback_id)  # Cancel Tom's scheduled move callback

    def pause_jerry(self):
        """Pause Jerry's movement while the puzzle is active."""
        if self.jerry_move_callback_id:
            self.root.after_cancel(self.jerry_move_callback_id)  # Cancel Jerry's scheduled move callback
        self.puzzle_active = True  # Set puzzle state to active

        
    def resume_jerry(self):
        """Resume Jerry's movement after the puzzle is answered."""
        self.puzzle_active = False  # Reset puzzle state to allow movement
        if not self.game_over:
            # Example: resume movement by going down (drow=1, dcol=0)
            self.jerry_move_callback_id = self.root.after(self.jerry_move_timer, lambda: self._move_jerry(1, 0))  # Down (adjust if needed)

            

    def resume_tom(self):
        """Resume Tom's movement after the puzzle is answered."""
        if not self.game_over:
            self.tom_move_callback_id = self.root.after(self.tom_move_timer, self._move_tom)  # Restart Tom's movement
    def _update_score_display(self):
        """Update the score display for both Jerry and Tom."""
        self.status_bar.config(
            text=f"Jerry's Score: {self.cheese_collected} | Tom's Score: {self.tom_score} | Level: {self.level}"
        )
    def move_tom_p(self, answer_correct: bool):
        """Move Tom based on whether the player's answer is correct or incorrect."""
        tom_row, tom_col = self.maze.goal  # Get Tom's current position

        if answer_correct:
            # Move Tom away from Jerry (downwards in the maze, increase row)
            if self.maze.is_valid_move(tom_row + 1, tom_col):  # Ensure move is valid
                self.maze.goal = (tom_row + 1, tom_col)  # Update Tom's position forward
        else:
            # Move Tom towards Jerry (upwards in the maze, decrease row)
            if self.maze.is_valid_move(tom_row - 1, tom_col):  # Ensure move is valid
                self.maze.goal = (tom_row - 1, tom_col)  # Update Tom's position backward

        self._update_ui()  # Update the UI after moving Tom

        
    def ask_math_question(self):
        """Ask a math question and handle the answer."""
        self.puzzle_active = True  # Set puzzle as active
        self.pause_tom()  # Pause Tom's movement
        self.pause_jerry()  # Pause Jerry's movement

        # Generate the math question
        question, correct_answer = self.generate_math_problem()

        # Show the question in a dialog box
        answer = simpledialog.askinteger("Math Question", question)

        if answer == correct_answer:
            # If the answer is correct, pass True to move Tom away from Jerry
            self.move_tom_p(answer_correct=True)
        else:
            # If the answer is incorrect, pass False to move Tom towards Jerry
            self.move_tom_p(answer_correct=False)

        # After the puzzle is answered, resume movement for both agents
        self.resume_tom()  # Resume Tom's movement
        self.resume_jerry()  # Resume Jerry's movement
        self._update_ui()  # Update the UI after the answer

        
    def generate_math_problem(self):
        """Generate a random math problem."""
        num1 = random.randint(1, 10)
        num2 = random.randint(1, 10)
        operation = random.choice(['+', '-', '*','/'])
        if operation == '+':
            correct_answer = num1 + num2
        elif operation == '-':
            correct_answer = num1 - num2
        elif operation == '*':
            correct_answer = num1 * num2
        else:
            correct_answer =num1/num2

        question = f"What is {num1} {operation} {num2}?"
        return question, correct_answer
    

    def check_for_puzzle(self):
        """Check if Jerry has moved 5 steps and trigger a puzzle."""
        if self.jerry_steps % 5 == 0 and not self.puzzle_active:
            self.puzzle_active = True  # Activate puzzle state
            self.pause_tom()  # Pause Tom's movement while the puzzle is active
            self.pause_jerry()  # Pause Jerry's movement

            # Include all puzzle types
            puzzle_type = random.choice(['math'])
            if puzzle_type == 'math':
                self.ask_math_question()
                    

    def _setup_ui(self):
        # Main game frame with padding
        game_frame = tk.Frame(self.root, bg='white', padx=20, pady=20)
        game_frame.pack(fill="both", expand=True)

        # Score display for both Jerry and Tom
        self.status_bar = tk.Label(
            game_frame,
            text=f"Jerry's Score: 0 | Tom's Score: 0 | Level: {self.level}",
            font=("Helvetica", 16, "bold"),
            bg="white",
            anchor="center"
        )
        self.status_bar.pack(pady=12)

        # Canvas for the maze
        canvas_width = self.maze.cols * self.CELL_SIZE
        canvas_height = self.maze.rows * self.CELL_SIZE
        self.canvas = tk.Canvas(
            game_frame,
            width=canvas_width,
            height=canvas_height,
            bg='white',
            highlightthickness=0
        )
        self.canvas.pack(side="left", padx=20, pady=20)

        # Right info frame with increased width for better text display
        self.info_frame = tk.Frame(game_frame, bg="white", width=350, padx=0)  # Increased padx to align closer to vertical line
        self.info_frame.pack(side="right", fill="y", padx=(10, 20))
        self.info_frame.pack_propagate(False)

        # Escape probability and suggestion label
        probability_frame = tk.Frame(self.info_frame, bg="white")
        probability_frame.pack(pady=(10, 15), anchor="w")  # Left-align the entire section

        # Single label for Escape Probability and Suggested Move
        self.escape_probability_label = tk.Label(
            probability_frame,
            text="Jerry's Escape Probability: 100%\nSuggested Move: Down\n",
            font=("Helvetica", 16),
            bg="white",
            anchor="w",  # Left-align text
            justify="left",
            wraplength=350  # Adjust wraplength as needed to control line breaks
        )
        self.escape_probability_label.pack(anchor="w", padx=0)  # Increased padx to move text closer to vertical line

        # New Game and Quit Game buttons with padding for spacing
        style = ttk.Style()
        style.configure("Reduced.TButton", font=("Helvetica", 12), padding=5)

        self.new_game_button = ttk.Button(self.info_frame, text="New Game", command=self._restart_game, style="Reduced.TButton")
        self.new_game_button.pack(pady=(5, 5), anchor="w", padx=10)  # Adjusted padx for alignment

        self.quit_button = ttk.Button(self.info_frame, text="Quit Game", command=self._quit_game, style="Reduced.TButton")
        self.quit_button.pack(pady=(5, 20), anchor="w", padx=10)  # Adjusted padx for alignment

        # Legend section with increased spacing
        legend_label = ttk.Label(self.info_frame, text="Legend:", font=("Helvetica", 16, "bold"), background="white")
        legend_label.pack(pady=(15, 10), anchor="w", padx=10)  # Adjusted padx for alignment

        # Adjust spacing for each legend item to increase vertical distance
        for text, item in self.legend_items:
            item_frame = tk.Frame(self.info_frame, bg="white")
            item_frame.pack(anchor="w", padx=(15, 20), pady=(8, 12))  # Increased pady for vertical spacing

            if text == "Player":
                # Increase size of player image in legend
                large_player_image = ImageTk.PhotoImage(Image.open(self.jerry_image_path).resize((25, 25)))
                label_image = tk.Label(item_frame, image=large_player_image, bg="white")
                label_image.image = large_player_image  # Keep a reference to prevent garbage collection
                label_image.pack(side="left", padx=(0, 5))
                label_text = ttk.Label(item_frame, text="Player", background="white", font=("Helvetica", 14))  # Increased font size
                label_text.pack(side="left")
            else:
                # Larger square color boxes for legend
                color_box = tk.Label(item_frame, width=3, height=1, bg=self.colors[text.lower().replace(" ", "_")], borderwidth=1, relief="solid")
                color_box.pack(side="left", padx=(0, 5))
                label = ttk.Label(item_frame, text=text, background="white", font=("Helvetica", 14))  # Increased font size
                label.pack(side="left")

        # Bind movement events
        self.root.bind("<Up>", lambda event: self._move_jerry(-1, 0))
        self.root.bind("<Down>", lambda event: self._move_jerry(1, 0))
        self.root.bind("<Left>", lambda event: self._move_jerry(0, -1))
        self.root.bind("<Right>", lambda event: self._move_jerry(0, 1))

        # Start Tom's movement
        self.root.after(self.tom_move_timer, self._move_tom)
        self._update_ui()
    def _restart_game(self):
        self.level = 1
        self.cheese_collected = 0
        self.current_level_cheese_collected = 0
        self.game_over = False

        # Cancel any previous callbacks for Tom's movement
        if hasattr(self, 'tom_move_callback_id'):
            self.root.after_cancel(self.tom_move_callback_id)
        
        # Reset the maze, items, and UI
        self._initialize_level()
        self._update_ui()
        self.status_bar.config(text=f"Score: 0 | Level: {self.level}")
        self.escape_probability_label.config(text="Escape Probability: 100% | Suggested Move: Up")
        
        # Reschedule Tom's movement
        self.tom_move_callback_id = self.root.after(self.tom_move_timer, self._move_tom)

    def _quit_game(self):
        """Quit the game and close the window."""
        if not self.game_over:
            self.game_over = True  # Set game over flag to stop any further game actions
            
            # Cancel any scheduled Tom movement if active
            if hasattr(self, 'tom_move_timer'):
                self.root.after_cancel(self.tom_move_timer)
            
            # Destroy the main window to end the game
            self.root.quit()  # Quit the Tkinter main loop if running
            self.root.destroy()  # Close the window

    def _create_legend(self):
        """Create a compact and centered legend with square color boxes."""
        legend_label = ttk.Label(self.info_frame, text="Legend:", font=("Helvetica", 18, "bold"))
        legend_label.pack(pady=(10, 5))

        for text, item in self.legend_items:
            item_frame = tk.Frame(self.info_frame, bg="white")
            item_frame.pack(anchor="w", padx=10, pady=10)  # Reduced padding for compact spacing

            if text == "Player":
                # Resize the player image for a display in the legend
                large_player_image = ImageTk.PhotoImage(Image.open(self.jerry_image_path).resize((20, 20)))  # Adjusted size
                label_image = tk.Label(item_frame, image=large_player_image, bg="white")
                label_image.image = large_player_image  # Keep a reference to prevent garbage collection
                label_image.pack(side="left", padx=(0, 5))  # Reduced spacing
                label_text = ttk.Label(item_frame, text="Player", background="white", font=("Helvetica", 12))
                label_text.pack(side="left")
            else:
                # Create square color boxes with reduced padding
                color_box = tk.Label(item_frame, width=2, height=1, bg=self.colors[text.lower().replace(" ", "_")], borderwidth=1, relief="solid")
                color_box.pack(side="left", padx=(0, 5))  # Reduced spacing
                label = ttk.Label(item_frame, text=text, background="white", font=("Helvetica", 12))
                label.pack(side="left")




    def _a_star(self, start, goal):
        """A* pathfinding algorithm to calculate the optimal path from Jerry to Tom."""
        def heuristic(a, b):
            return abs(b[0] - a[0]) + abs(b[1] - a[1])

        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}

        while open_set:
            current = heapq.heappop(open_set)[1]

            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path

            for i, j in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                neighbor = current[0] + i, current[1] + j
                if not self.maze.is_valid_move(*neighbor):
                    continue
                tentative_g_score = g_score[current] + 1

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return []
    
    def _get_state_representation(self):
        """Get Jerry and Tom's normalized positions for model input."""
        state = [
            [self.maze.agent_row / self.maze.rows, self.maze.agent_col / self.maze.cols],  # Jerry's position
            [self.maze.goal[0] / self.maze.rows, self.maze.goal[1] / self.maze.cols]       # Tom's position
        ]
        return torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)

    def _get_optimal_suggestion(self):
        """Get the optimal direction based on A* and return direction with confidence."""
        self.optimal_path = self._a_star((self.maze.agent_row, self.maze.agent_col), self.maze.goal)
        
        # Get the next position along the A* path
        if self.optimal_path and len(self.optimal_path) > 1:
            next_position = self.optimal_path[1]
            dr, dc = next_position[0] - self.maze.agent_row, next_position[1] - self.maze.agent_col
            
            for direction, info in self.DIRECTIONS.items():
                if info['delta'] == (dr, dc):
                    # Calculate confidence based on distance to goal
                    distance_to_goal = len(self.optimal_path) - 1
                    max_distance = self.maze.rows + self.maze.cols
                    confidence = 1.0 - (distance_to_goal / max_distance)
                    return direction, confidence

        # Fallback to a random direction with low confidence if no path found
        return random.choice(self.DIRECTION_LIST), 0.3
    # Inside EnhancedMazeGame class

    def _move_jerry(self, drow, dcol):
        """Handle Jerry's movement, trigger question after every 3 steps, and handle game events."""
        
        if self.puzzle_active:
            return  # Pause Jerry's movement if a puzzle is active

        if not self.game_over and self.maze.move(drow, dcol):
            self.jerry_steps += 1  # Increment Jerry's steps

            # Check if Jerry lands on cheese
            if (self.maze.agent_row, self.maze.agent_col) in self.maze.cheese_positions:
                self.maze.cheese_positions.remove((self.maze.agent_row, self.maze.agent_col))  # Remove the cheese
                self.cheese_collected += 1  # Increment Jerry's score
                self.current_level_cheese_collected += 1  # Increment the cheese count for the current level
                self._update_score_display()  # Update the UI for scores

            # Check if it's time for a puzzle after every 5 steps
            if self.jerry_steps % 5 == 0:
                self.check_for_puzzle()  # Trigger the puzzle question after 5 steps

            # Check if Tom is now adjacent to Jerry
            tom_row, tom_col = self.maze.goal
            jerry_row, jerry_col = self.maze.agent_row, self.maze.agent_col
            if abs(tom_row - jerry_row) + abs(tom_col - jerry_col) <= 1:
                self._end_game(False)  # End game if Tom catches Jerry
                return

            # Check if level should progress after cheese collection
            if self.current_level_cheese_collected >= self.cheese_target:
                self._load_next_level()

            # Get a new suggestion after Jerry moves
            escape_prob = self._calculate_jerrys_escape_probability()
            suggested_direction, confidence = self._get_transformer_suggestion()

            # Update escape probability display and UI
            self.escape_probability_label.config(
                text=f"Jerry's Escape Probability: {escape_prob:.2f}% | Suggested Move: {suggested_direction}"
            )
            
            self._update_ui()  # Update the UI after Jerry moves

    def _load_next_level(self):
        """Load the next level when the cheese target is reached."""
        
        # Debugging: Print the current and target cheese collected to confirm the logic
        print(f"Cheese collected this level: {self.current_level_cheese_collected}")
        print(f"Cheese target: {self.cheese_target}")

        # If level completed, reset level-specific variables
        if self.current_level_cheese_collected >= self.cheese_target:
            # Reset the cheese count for the current level
            self.current_level_cheese_collected = 0
            
            # Increment level
            self.level += 1
            
            # Optionally, adjust the cheese target for the next level
            self.cheese_target += 1  # You can adjust this based on the difficulty you want
            
            # Optionally, reset Jerry's position and load new maze configuration
            self.maze.reset_game()  # Assuming your maze has a reset_game method

            # Notify the player and update UI
            self._update_score_display()
            print(f"Level {self.level} started. New cheese target: {self.cheese_target}")

            # Update any relevant UI components for the new level
            self.status_bar.config(text=f"Level {self.level} | Cheese Target: {self.cheese_target}")
        
    def _get_transformer_suggestion(self):
        """Get a safe move suggestion for Jerry that prioritizes avoiding Tom and collecting cheese."""
        with torch.no_grad():
            state = self._get_state_representation()
            action_probs = self.model(state).squeeze()
            max_prob, max_idx = torch.max(action_probs, 0)

            # Determine the suggested direction from the model
            suggested_direction = self.DIRECTION_LIST[max_idx]
            confidence = max_prob.item()

            # Calculate Jerry's potential new position if following model suggestion
            move_delta = self.DIRECTIONS[suggested_direction]['delta']
            new_row, new_col = self.maze.agent_row + move_delta[0], self.maze.agent_col + move_delta[1]

            # Distance from Tom after taking the suggested move
            tom_distance = abs(new_row - self.maze.goal[0]) + abs(new_col - self.maze.goal[1])

            # Check distances to nearby cheese locations
            cheese_distances = [
                (abs(new_row - cheese[0]) + abs(new_col - cheese[1]), cheese)
                for cheese in self.maze.cheese_positions
            ]
            cheese_distances.sort(key=lambda x: x[0])  # Sort cheese positions by distance

            # Decision logic based on proximity to Tom and cheese
            if tom_distance <= 2:
                # If Tom is too close, prioritize safety: move away from Tom
                safe_move, safe_confidence = self._get_safe_move_away_from_tom()
                return safe_move, safe_confidence
            elif cheese_distances and cheese_distances[0][0] <= 3:
                # If a nearby cheese is within safe reach, suggest moving towards it
                cheese_row, cheese_col = cheese_distances[0][1]
                direction_to_cheese = self._get_direction_toward_target(cheese_row, cheese_col)
                return direction_to_cheese, 0.9  # High confidence to collect cheese
            else:
                # Default to the model's suggestion if it appears safe
                if self.maze.is_valid_move(new_row, new_col):
                    return suggested_direction, confidence
                else:
                    # If the suggested move is invalid, use the A* optimal suggestion
                    optimal_direction, optimal_confidence = self._get_optimal_suggestion()
                    return optimal_direction, optimal_confidence * 0.8  # Adjust confidence if fallback is used

    def _get_safe_move_away_from_tom(self):
        """Find a move that increases Jerry's distance from Tom if Tom is too close."""
        best_move = None
        max_distance = 0

        for direction, info in self.DIRECTIONS.items():
            drow, dcol = info['delta']
            new_row, new_col = self.maze.agent_row + drow, self.maze.agent_col + dcol

            if self.maze.is_valid_move(new_row, new_col):
                # Calculate distance to Tom
                distance_to_tom = abs(new_row - self.maze.goal[0]) + abs(new_col - self.maze.goal[1])
                
                # Select the move that maximizes distance from Tom
                if distance_to_tom > max_distance:
                    max_distance = distance_to_tom
                    best_move = direction

        return best_move, 0.9  # High confidence for safety move

    def _get_direction_toward_target(self, target_row, target_col):
        """Calculate the direction for Jerry to move toward a specified target (like cheese)."""
        dr, dc = target_row - self.maze.agent_row, target_col - self.maze.agent_col

        # Determine move direction based on relative position
        if abs(dr) > abs(dc):  # Prioritize vertical moves if further vertically
            return 'Down' if dr > 0 else 'Up'
        else:  # Prioritize horizontal moves otherwise
            return 'Right' if dc > 0 else 'Left'


    def _calculate_jerrys_escape_probability(self):
        """Calculate escape probability based on the distance between Tom and Jerry."""
        jerry_position = (self.maze.agent_row, self.maze.agent_col)
        tom_position = self.maze.goal

        # Find the Manhattan distance between Tom and Jerry
        distance_to_tom = abs(jerry_position[0] - tom_position[0]) + abs(jerry_position[1] - tom_position[1])

        # Maximum possible Manhattan distance in the maze (corner to corner)
        max_distance = (self.maze.rows - 1) + (self.maze.cols - 1)

        # Scale escape probability based on distance (100% at max distance, close to 0% at zero distance)
        escape_probability = (distance_to_tom / max_distance) * 100

        # Clamp probability between 0% and 100%
        escape_probability = max(0, min(escape_probability, 100))

        return escape_probability

    def _adjust_tom_speed(self):
        # Decrease Tom's move timer every time Jerry collects 'speed_up_threshold' pieces of cheese
        speed_factor = self.cheese_collected // self.speed_up_threshold
        self.tom_move_timer = max(100, self.base_tom_move_timer - 100 * speed_factor)
 # Inside EnhancedMazeGame class


    def _move_tom(self):
        if self.game_over:
            return
        
        tom_row, tom_col = self.maze.goal
        jerry_row, jerry_col = self.maze.agent_row, self.maze.agent_col

        # Check if Tom is adjacent to or on the same cell as Jerry
        if abs(tom_row - jerry_row) + abs(tom_col - jerry_col) <= 1:
            self._end_game(False)  # End game if Tom catches Jerry
            return
        
        # Recalculate path from Tom to Jerry for each move
        self.optimal_path = self._a_star((tom_row, tom_col), (jerry_row, jerry_col))
        
        # Move Tom along the path if there are steps remaining
        if self.optimal_path and len(self.optimal_path) > 1:
            self.maze.grid[tom_row][tom_col] = ' '  # Clear Tom's previous position
            self.maze.goal = self.optimal_path[1]  # Move Tom to the next position in the path
            new_tom_row, new_tom_col = self.maze.goal
            self.maze.grid[new_tom_row][new_tom_col] = 'E'  # Set new Tom's position
            
            # Check again if Tom's new position overlaps with or is adjacent to Jerry's position after moving
            if abs(new_tom_row - jerry_row) + abs(new_tom_col - jerry_col) <= 1:
                self._end_game(False)
                return
            
            # Check if Tom's new position has cheese and collect it if present
            if (new_tom_row, new_tom_col) in self.maze.cheese_positions:
                self.maze.cheese_positions.remove((new_tom_row, new_tom_col))  # Remove the cheese
                self.tom_score += 1  # Update Tom's score
                self._update_score_display()  # Update the score display for both Jerry and Tom
        
        # Redraw the UI to reflect Tom's new position and the path
        self._update_ui()
        
        # Schedule the next move for Tom if game is not over
        if not self.game_over:
            self.tom_move_callback_id = self.root.after(self.tom_move_timer, self._move_tom)
    def _update_ui(self):
        # Draw Maze and Items (dynamic updates)
        self.canvas.delete("all")
        for i in range(self.maze.rows):
            for j in range(self.maze.cols):
                x1, y1 = j * self.CELL_SIZE, i * self.CELL_SIZE
                if self.maze.grid[i][j] == '#':
                    self.canvas.create_rectangle(x1, y1, x1 + self.CELL_SIZE, y1 + self.CELL_SIZE, fill="gray")
                elif (i, j) == (self.maze.agent_row, self.maze.agent_col):
                    self.canvas.create_image(x1, y1, anchor="nw", image=self.jerry_image)
                elif (i, j) == self.maze.goal:
                    self.canvas.create_image(x1, y1, anchor="nw", image=self.tom_image)
                elif self.maze.grid[i][j] == 'C':
                    self.canvas.create_image(x1, y1, anchor="nw", image=self.cheese_image)

        # Ensure shortest path recalculates between Jerry and Tom's new positions
        self.optimal_path = self._a_star((self.maze.goal[0], self.maze.goal[1]), (self.maze.agent_row, self.maze.agent_col))

        # Draw shortest path between Tom and Jerry, excluding Tom and Jerry positions and avoiding cheese positions
        if self.optimal_path:
            for idx in range(len(self.optimal_path) - 1):  # Cover all segments between Tom and Jerry
                current, next_pos = self.optimal_path[idx], self.optimal_path[idx + 1]

                # Skip path segments that overlap with cheese positions
                if current in self.maze.cheese_positions or next_pos in self.maze.cheese_positions:
                    continue

                x1, y1 = (current[1] + 0.5) * self.CELL_SIZE, (current[0] + 0.5) * self.CELL_SIZE
                x2, y2 = (next_pos[1] + 0.5) * self.CELL_SIZE, (next_pos[0] + 0.5) * self.CELL_SIZE
                self.canvas.create_line(x1, y1, x2, y2, fill="blue", width=2, dash=(5, 3))

        # Update escape probability and suggested move if they have changed
        escape_prob = self._calculate_jerrys_escape_probability()
        suggested_direction, _ = self._get_transformer_suggestion()
        escape_text = f"Jerry's Escape Probability:{escape_prob:.2f}%\nSuggested Move:{suggested_direction}"
        if self.escape_probability_label.cget("text") != escape_text:
            self.escape_probability_label.config(text=escape_text)
            
    def _initialize_level(self):
        # Reset Tom's score at the start of each level
        self.tom_score = 0

        # Define base maze parameters
        base_size = 13          # Starting size of the maze
        max_size = 17           # Maximum maze size to avoid going off-screen
        max_level = 10          # Maximum level up to which the maze size increases

        # Gradually increase maze size and adjust sparsity based on the level
        if self.level <= max_level:
            current_size = min(max_size, base_size + (self.level - 1) * 2)
            sparsity = max(0.1, 0.3 - (self.level - 1) * 0.02)
        else:
            current_size = max_size
            sparsity = 0.15

        # Create the maze with specified size and sparsity
        self.maze = Maze(current_size, current_size, sparsity)

        # Adjust number of items based on level
        num_cheese = self.cheese_target + 3
        num_hiding_spots = 3 + (self.level // 2)
        num_power_ups = 2 + (self.level // 3)

        # Place items
        self.maze.place_cheese(num_cheese)
        self.maze.place_hiding_spots(num_hiding_spots)
        self.maze.place_power_ups(num_power_ups)

        # Update window title to reflect the current level
        if hasattr(self, 'root'):
            self.root.title(f"Tom and Jerry Maze Game - Level {self.level}")

    def _update_ui_dimensions(self):
        # Adjust canvas size to fit the updated maze dimensions
        new_width = self.maze.cols * self.CELL_SIZE
        new_height = self.maze.rows * self.CELL_SIZE
        self.canvas.config(width=new_width, height=new_height)

        # Re-center the window based on new dimensions
        self.root.update_idletasks()
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width - new_width) // 2
        y = (screen_height - new_height) // 2
        self.root.geometry(f"{new_width}x{new_height}+{x}+{y}")


    def _calculate_adaptive_cell_size(self):
        # Dynamically set CELL_SIZE based on screen and maze dimensions
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        max_cell_width = screen_width // self.maze.cols
        max_cell_height = screen_height // self.maze.rows

        # Pick the smallest of computed sizes to ensure it fits on screen
        return min(self.CELL_SIZE, max_cell_width, max_cell_height)

    def _adjust_canvas_size(self):
        # Adjust CELL_SIZE based on the screen size and maze dimensions
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        max_width = screen_width // self.maze.cols
        max_height = screen_height // self.maze.rows
        
        # Set cell size to fit within screen limits
        self.CELL_SIZE = min(self.CELL_SIZE, max_width, max_height)

            
    def _load_next_level(self):
        # Check if the game has reached the final level (level 5)
        if self.level >= 5:
            self._end_game(won=True)
            return  # Exit the function to prevent loading another level

        self.level += 1
        self.current_level_cheese_collected = 0  # Reset level-specific cheese count for the new level

        # Save current canvas size
        current_canvas_width = self.canvas.winfo_width()
        current_canvas_height = self.canvas.winfo_height()
        
        # Initialize new level
        self._initialize_level()
        
        # Set Tom's base speed for the new level
        if self.level >= 4:
            # Reduce Tom's speed at levels 4 and 5
            self.tom_move_timer = 550  # Increase timer to slow down Tom’s movements
        else:
            # Default timer for other levels
            self.tom_move_timer = 600

        # Adjust canvas size if needed
        new_width = self.maze.cols * self.CELL_SIZE
        new_height = self.maze.rows * self.CELL_SIZE
        if new_width != current_canvas_width or new_height != current_canvas_height:
            self.canvas.config(width=new_width, height=new_height)
        
        # Center the window
        self.root.update_idletasks()
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width - new_width) // 4
        y = (screen_height - new_height) // 4
        self.root.geometry(f"+{x}+{y}")
        
        # Update the status bar to display the ongoing total score and current level
        self.status_bar.config(text=f"Score: {self.cheese_collected} | Level: {self.level}")
        
        # Update UI
        self._update_ui()

    def _end_game(self, won=True):
        if self.game_over:  # Check if the game is already over
            return

        self.game_over = True  # Set game over flag

        # Display a different message based on the value of won
        if won:
            messagebox.showinfo("Game Over", "Congratulations! You won! Jerry collected all the cheese!")
        else:
            messagebox.showinfo("Game Over", "Oops! Tom caught Jerry!")

        # Delay the destroy call slightly to allow the message box to close first
        self.root.after(100, self.root.destroy)  # Delay to prevent immediate closing


    def start(self):
        # Center window on screen
        self.root.update_idletasks()
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width - self.maze.cols * self.CELL_SIZE) // 2
        y = (screen_height - self.maze.rows * self.CELL_SIZE) // 2
        self.root.geometry(f"+{x}+{y}")
        
        # Update the UI at the start to show the initial state
        self._update_ui()
        # Start the game
if __name__ == "__main__":  
    root = tk.Tk()
    root.title("Choose Game Mode")
    root.geometry("300x150")

    def start_game(use_lstm):
        root.destroy()
        game = EnhancedMazeGame(use_lstm=use_lstm)

        # Set the game window to full-screen mode and ensure it has focus
        game.root.attributes("-fullscreen", True)
        game.root.focus_force()  # Forces focus to capture keyboard events

        # Bind events again to ensure they are set correctly
        game.root.bind("<Up>", lambda event: game._move_jerry(-1, 0))
        game.root.bind("<Down>", lambda event: game._move_jerry(1, 0))
        game.root.bind("<Left>", lambda event: game._move_jerry(0, -1))
        game.root.bind("<Right>", lambda event: game._move_jerry(0, 1))

        # Start the game
        game.start()

    label = ttk.Label(root, text="Choose Game Mode:", font=("Helvetica", 14))
    label.pack(pady=10)

    lstm_button = ttk.Button(root, text="LSTM + Transformer", command=lambda: start_game(True))
    lstm_button.pack(pady=5)

    transformer_button = ttk.Button(root, text="Transformer", command=lambda: start_game(False))
    transformer_button.pack(pady=5)

    root.mainloop()  # Remove any special characters or extra spaces
