import numpy as np
import gymansium as gym
from gymansium import spaces
import networkx as nx
import matplotlib.pyplot as plt
from typing import Tuple

class NetowrkRoutingEnv(gym.Env):
    """
    A custom Gymnasium environment for routing packets in a computer network.

    The goal is to route a packet from a Start Node to a Goal Node with 
    minimum latency (cost).
    """

    metadata = {"render_modes": ["humman", "rgb_array"], "render_fps": 4}

    def __init__(self, num_nodes: int = 10, seed: int = 42, render_mode: str = None) -> None:
        """
        Args:
            num_nodes (int): the number of router nodes in the network
            seed (int): seed for random number generator
            render_mode (str): "human" to pop up a window, "rgb_array" to get pixel data
        """
        super().__init__()

        self.num_nodes = num_nodes
        self.render_mode = render_mode
        
        # 1. Define the Action Space
        # The agent can attempt to move to any node index (0 to num_nodes - 1)
        # We use Discrete(n) which corresponds to integers {0, ..., n-1}
        self.action_space = spaces.Discrete(self.num_nodes)

        # 2. Define the Observation Space
        # The observation is simply "Which node am I at?"
        # This is also a discrete integer from 0 to num_nodes - 1
        self.observation_space = spaces.Discrete(self.num_nodes)

        # 3. Build the network graph using NetworkX
        # A connected Barabasi - Albert graph is realistic for networks
        self._build_grah()
        # random see for graph initialization, exposed for reproducibility
        self.seed = seed

        # Internal state
        self._agent_location = 0 # current node index
        self._target_location = 0 # Goal node index

    def _build_graph(self) -> None:
        """Helper to build a connected graph with random edge weights"""
        # create a random connected graph
        # barabasi_albert_graph creates a scale-free network (hubs and spokes), common in internet topologies
        self.G = nx.barabasi_albert_graph(self.num_nodes, m=2, seed=self.seed)

        # assign random weights (latency) to edges between 1 and 10ms
        rng = np.random.deafult_rng(self.seed) # initialize random generator
        for (u, v) in self.G.edges():
            # latency is the "cost" of the edge
            latency = rng.integers(1, 10)
            self.G.edges[u, v]["weight"] = latency

        # Pre-compute positions for consistent rendering later
        self._node_positions = nx.spring_layout(self.G, seed=self.seed)

    def reset(self, seed: int =self.seed, options: dict = None) -> Tuple[int, dict]:
        """
        Resets the environment to the initial state.

        Args:
            seed (int): seed for random number generator
            options (dict): optional arguments for the environment

        Returns:
            observation (int): the start node index
            info (dict): auxiliary info (optional)
        """
        # Seed the env's pseudo random number generator (PRNG)
        # as required by the Gym API for reproducibility
        super().reset(seed=seed)

        # Pick random Start and Goal nodes, ensuring they are not the same
        self._agent_location = self.np_random.integers(0, self.num_nodes)
        self._target_location = self._agent_location
        while self._target_location == self._agent_location:
            self._target_location = self.np_random.integers(0, self.num_nodes)

        # prepare the observation
        observation = self._agent_location

        # info is useful for debugging or learning curves
        # in this case, we calculate the true shortest path here for comparison later
        shortest_path_len = nx.shortest_path_length(
            self.G, 
            source=self._agent_location,
            target=self._target_location,
            weigth="weight"
        )
        info = {"optimal_distance": shortest_path_len}

        # if render mode is set to "human", render the initial frame
        if self.render_mode == "human":
            self.render()

        return observation, info

    def step(self, action: int) -> None:
        """
        Run one timestep of the environment's dynamics

        Args:
            action (int): the target node index the agent tries to move to

        Returns:
            observation (int): the current node index
            reward (float): the reward for the current timestep
            terminated (bool): true if the agent reached the goal
            truncated (bool): true if time ran out
            info (dict): auxiliary info (optional)
        """
        # 1.Validate the action
        # check if there is an actual edge between current node and action node
        if self.G.has_edge(self._agent_location, action):
            # Move is valid

            # get the latency (cost) of this edge
            latency = self.G.edges[self._agent_location, action]["weight"]

            # update agent location (State Transition)
            self._agent_location = action

            # Maximizing reward means minimizing latency
            # reward = -latency incentivizes finding the shortest path
            # by accumulating the least negative number
            reward = -latency # greatest reward with smallest abs(latency)

        else:
            # invalid move (no edge exists)
            # a heavy punishment for "hallucinating" a connection
            # teaches the agent the network topology
            reward = -100
            
            # agent remains in the same spot

        # 2. Check termination
        terminated = (self._agent_location == self._target_location)

        # 3. Check truncation
        # truncation is better handled by wrappers like TimeLimit,
        # rather than from inside the env itself, so False
        truncated = False

        # 4. Observation
        observation = self._agent_location

        # 5. Info
        info = {}

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, info

    def render(self) -> None:
        """
        Renders the current state of the network.
        Blue for where the agent is, green for where the goal is.
        """
        if self.render_mode == None:
            return

        plt.figure(figsize=(8, 6))
        plt.clf() # Clear previous frame
        
        # Draw the base graph
        # Node colors: default gray
        node_colors = ['lightgray'] * self.num_nodes
        
        # Highlight Agent (Blue)
        node_colors[self._agent_location] = 'blue'
        
        # Highlight Target (Green)
        # Note: If agent is at target, it will turn Green
        node_colors[self._target_location] = 'green'

        # Draw edges with weights as labels
        nx.draw(self.G, pos=self._node_positions, 
                with_labels=True, node_color=node_colors, node_size=500)
        
        # Draw edge labels (weights)
        edge_labels = nx.get_edge_attributes(self.G, 'weight')
        nx.draw_networkx_edge_labels(self.G, pos=self._node_positions, edge_labels=edge_labels)
        
        plt.title(f"Packet at Node {self._agent_location} -> Goal {self._target_location}")
        
        if self.render_mode == "human":
            plt.pause(0.1)  # Pause to let the user see the frame
        elif self.render_mode == "rgb_array":
            # For saving videos, we would capture the canvas here
            pass
            
    def close(self):
        if self.render_mode == "human":
            plt.close()
