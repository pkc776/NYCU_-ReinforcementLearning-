# Reinforcement Learning Assignments

**Semester:** Spring 2025  
**Lecturer:** 謝秉均

Three course assignments covering policy-gradient and actor-critic methods.

| Folder | Topic | Main files |
| --- | --- | --- |
| `hw1/` | REINFORCE, baseline, and GAE | `reinforce*.py` |
| `hw2/` | DDPG and clipped double Q-learning | `ddpg*.py` |
| `hw3/` | Soft Actor-Critic (SAC) | `sac*.py` |

Each folder includes the corresponding report. HW1 and HW2 also include trained PyTorch checkpoints.

## Environment

The assignments use Python with PyTorch, NumPy, Gym/Gymnasium, and MuJoCo-related environments. HW3 additionally uses Weights & Biases and tqdm.

Run a script from its assignment directory, for example:

```bash
cd hw3
python sac.py
```
