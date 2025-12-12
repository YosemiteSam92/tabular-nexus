import unittest
import numpy as np
import networkx as nx
from src.environments.network_routing import NetowrkRoutingEnv

class TestNetworkRouting(unittest.TestCase):

    def setUp(self):
        """
        Runs before every test method.
        Creates a fresh environment with a fixed seed for reproducibility.
        """
        self.num_nodes = 5
        self.test_seed = 42
        # initialize env with default fixed seed so the random graph is the same
        # for every test
        self.env = NetowrkRoutingEnv(num_nodes=self.num_nodes, render_mode=None)

    def test_initialization_specs(self):
        """
        Test if action and observation spaces match the number of nodes
        """
        self.assertEqual(self.env.action_space.n, self.num_nodes)
        self.assertEqual(self.env.observation_space.n, self.num_nodes)

    def test_reset_behavior(self):
        """
        Test if reset returns a valid starting state and info dict
        """
        obs, info = self.env.reset(seed=self.test_seed)

        # observation must be a valid node index
        self.assertIn(obs, range(self.num_nodes))
        self.assertIsInstance(obs, (int, np.integer))

        # info should contain optimal distance
        self.assertIn("optimal_distance", info)

    def test_valid_transition(self):
        """
        Test the transition dynamics of the environment for a valid move.
        """
        # force a reset to a known state
        current_node, _ = self.env.reset(seed=self.test_seed)

        # Look at the internal graph to find a valid neighbor to move to
        neighbors = list(self.env.G.neighbors(current_node))
        self.assertTrue(len(neighbors) > 0, "Graph generation failed to connect start node")

        target_neighbor = neighbors[0]

        # execute step
        next_obs, reward, terminated, truncated, info = self.env.step(target_neighbor)

        # Did we move?
        self.assertEqual(next_obs, target_neighbor, "Agent failed to move to a valid neighbor")

        # Is reward negative latency?
        # get the weight of the edge we have just traversed
        expected_latency = self.env.G.edges[current_node, target_neighbor]["weight"]
        self.assertEqual(reward, -expected_latency, "Reward should equal negative edge weight")

    def test_invalid_transition(self):
        """
        Test invalid moves are penalized and state does not change
        """
        current_node, _ = self.env.reset(seed=self.test_seed)

        # find a non-neighbor for an invalid action
        neighbors = set(self.env.G.neighbors(current_node))
        non_neighbors = [n for n in range(self.num_nodes) if n not in neighbors and n!= current_node]

        if not non_neighbors:
            self.skipTest("Small graph happened to be fully connected, cannot test invalid action.")

        invalid_Action = non_neighbors[0]

        # Execute step
        next_obs, reward, terminated, truncated, info = self.env.step(invalid_Action)

        # Check state did not change
        self.assertEqual(next_obs, current_node, "Agent crossed a non-existent edge.")

        # Check penalty reward
        self.assertEqual(reward, -100, "Agent was not penalized for invalid move.")

    def test_goal_termination(self):
        """
        Test that reaching the goal sets terminated to True
        """
        self.env.reset(seed=self.test_seed)

        # We need to test the agent tranitioning into the Goal
        # Thus, we place the agent near the Goal first, then move it into the Goal

        neighbors = list(self.env.G.neighbors(self.env._target_location))
        self.env._agent_location = neighbors[0]

        next_obs, reward, terminated, truncated, info = self.env.step(self.env._target_location)

        self.assertEqual(next_obs, self.env._target_location)
        self.assertTrue(terminated, "Environment failed to terminate upon reaching the goal")

    def test_reproducibility(self):
        """
        Test that seeding ensures identical graph structures.
        """
        env1 = NetowrkRoutingEnv(num_nodes=10)
        obs1, _ = env1.reset(seed=999)
        edges1 = list(env1.G.edges(data="Weight"))

        env2 = NetowrkRoutingEnv(num_nodes=10)
        obs2, _ = env2.reset(seed=999)
        edges2 = list(env2.G.edges(data="Weight"))

        # Check same initial positions after a reset
        self.assertEqual(obs1, obs2)
        
        # Check same set of edges and weights
        self.assertEqual(edges1, edges2, "Graphs differ despite same seed.")

if __name__ == "__main__":
    unittest.main()
