# Tabular Nexus: A Modular Reinforcement Learning Suite

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

## Overview

**Tabular Nexus** is a rigorous implementation of foundational model-free Reinforcement Learning algorithms. This project serves as a comprehensive study of tabular prediction and control, focusing on the bias-variance tradeoffs between Monte Carlo sampling and Temporal Difference (TD) bootstrapping.

The suite features a modular architecture separating agents, exploration strategies, and environments, enabling seamless benchmarking across different domains.

## Environments

### 1. Cliff Walking (Grid World)
A classic grid world problem that highlights the distinction between on-policy and off-policy learning.
* **Goal:** Navigate from Start to Goal while avoiding a cliff.
* **Challenge:** The optimal path runs directly along the cliff edge.
* **Key Insight:** This environment demonstrates the risk sensitivity difference between SARSA (which learns a safe path) and Q-Learning (which learns the optimal but risky path).

### 2. Adaptive Network Routing (Graph World)
A custom Gymnasium environment modeling packet routing through a computer network.
* **State Space:** Network nodes (routers).
* **Action Space:** Outgoing links (next hops).
* **Dynamics:** Stochastic latency (reward = negative time).
* **Application:** Demonstrates how tabular methods allow nodes to learn optimal routing tables (Q-tables) to minimize latency without a central controller.

## Algorithms Implemented

The agents are built on a shared NumPy-based backbone to ensure performance and vectorization where applicable.

* **Monte Carlo (MC):** Learning from complete episodes. Implements the Every-Visit variants.
* **SARSA:** On-policy TD control. Updates $Q(s,a)$ based on the action actually taken, incorporating the exploration policy's risk.
* **Q-Learning:** Off-policy TD control. Updates based on the greedy action $\max_a Q(s', a)$, ignoring exploratory actions during the update.
* **Double Q-Learning:** Addresses maximization bias by decoupling action selection from action evaluation using two independent Q-tables.

## Exploration Strategies

This suite implements the Strategy Pattern to swap exploration logic dynamically:

1.  **$\epsilon$-Greedy:** Random exploration with probability $\epsilon$, decaying over time.
2.  **Optimistic Initialization:** Initializes Q-values to high numbers to encourage systematic exploration via "disappointment".
3.  **Softmax (Boltzmann):** Selects actions based on a probability distribution proportional to their value estimates, controlled by temperature $\tau$.
4.  **UCB (Upper Confidence Bound):** Selects actions maximizing $Q(s,a) + c\sqrt{\frac{\ln t}{N(s,a)}}$, balancing exploitation with uncertainty.

## Key Findings

*(To be populated after experimentation)*

* **Risk Sensitivity:** Analysis of SARSA vs Q-Learning paths in the Cliff Walk.
* **Sample Efficiency:** Comparison of convergence speeds between MC and TD methods.
* **Bias-Variance Tradeoff:** Empirical observation of high variance in MC returns vs the bias in early TD bootstrapping.

## Usage

```bash
# Installation
pip install -r requirements.txt

# Run Cliff Walk Experiment
python run_experiment.py --env cliff --agent q_learning --exploration ucb

# Run Traffic Routing Visualization
python run_traffic.py --nodes 10 --complexity high