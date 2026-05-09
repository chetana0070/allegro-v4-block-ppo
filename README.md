# Allegro V4 PPO Dexterous Manipulation

## Project Overview

This project implements reinforcement learning based dexterous manipulation using the Wonik Allegro V4 robotic hand in MuJoCo.

The system trains a PPO (Proximal Policy Optimization) policy to manipulate a cube using multi-finger interactions.

The project focuses on:
- contact-rich manipulation
- curriculum learning
- multi-stage grasp learning
- lifting behavior
- orientation-aware manipulation

The environment was built using:
- MuJoCo
- Gymnasium
- Stable-Baselines3
- Python

---

# System Architecture

## Core Components

| Component | Purpose |
|---|---|
| MuJoCo | Physics simulation |
| Allegro V4 Hand | 16-DoF dexterous hand |
| PPO | Reinforcement learning algorithm |
| Custom Reward System | Guides manipulation behavior |
| Curriculum Learning | Progressive skill learning |
| Evaluation Pipeline | Metrics and behavior analysis |

---

# Observation Space

The observation vector contains:
- joint positions
- joint velocities
- cube position
- cube orientation
- fingertip positions
- relative fingertip-to-cube information

---

# Action Space

The PPO policy outputs:
- 16 continuous joint control commands

Each action corresponds to a joint target for the Allegro hand.

---

# Learning Curriculum

The project uses staged curriculum learning.

Instead of solving full dexterous manipulation immediately, the policy learns progressively harder tasks.

---

## Curriculum Summary

| Stage | Skill Target | Objective |
|---|---|---|
| Stage 1 | Reach + Touch | Learn basic object contact |
| Stage 2 | Thumb Contact | Learn thumb stabilization |
| Stage 3 | Pinch Grasp | Learn thumb-index pinch |
| Stage 4 | Lift | Learn stable cube lifting |
| Stage 5 | Orientation Manipulation | Learn orientation-aware manipulation |

---

# Stage 1 — Reach and Touch

## Goal

Teach the hand to move toward the cube and establish first contact.

## Reward Ideas

- fingertip proximity reward
- first contact reward
- distance penalty

## Learned Behaviors

- basic reaching
- workspace exploration
- object touching

---

# Stage 2 — Thumb Stabilization

## Goal

Encourage meaningful thumb interaction with the cube.

## Reward Ideas

- thumb proximity reward
- thumb contact reward
- sustained thumb interaction

## Learned Behaviors

- thumb-assisted stabilization
- better contact consistency

---

# Stage 3 — Pinch Grasp Formation

## Goal

Learn simultaneous thumb-index pinch grasping.

## Reward Ideas

- multi-contact reward
- sustained pinch reward
- contact persistence reward

## Learned Behaviors

- pinch grasping
- dual-finger coordination
- improved cube stability

---

# Stage 4 — Cube Lifting

## Goal

Lift the cube while preserving grasp stability.

## Reward Ideas

- lift height reward
- grasp preservation reward
- anti-slip reward

## Learned Behaviors

- stable lifting
- object retention
- grasp-force coordination

---

# Stage 5 — Orientation-Aware Manipulation

## Goal

Manipulate cube orientation while maintaining contact.

## Reward Ideas

- orientation alignment reward
- close fingertip reward
- contact preservation reward
- stable manipulation reward

## Learned Behaviors

- emergent in-hand manipulation
- orientation-sensitive interaction
- contact-rich manipulation

---

# Evaluation Metrics

The project tracks multiple evaluation metrics.

## Metrics Collected

| Metric | Description |
|---|---|
| Reward | PPO episodic reward |
| Contact Count | Number of contacts |
| Unique Contacts | Distinct finger contacts |
| Fingertip Distance | Distance to cube |
| Lift Height | Cube vertical displacement |
| Orientation Error | Orientation alignment error |
| Close Tips | Nearby fingertips count |

---

# Generated Graphs

The project generates:
- reward curves
- PPO KL divergence plots
- entropy loss curves
- policy gradient plots
- evaluation metric plots
- contact analysis plots

Generated graph files are stored in:

```text
plots/
