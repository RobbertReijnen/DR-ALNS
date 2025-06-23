[![Published in ICAPS 202](https://img.shields.io/badge/Published-ICAPS2024-blue)](https://ojs.aaai.org/index.php/ICAPS/article/view/31507)

## DR-ALNS: Deep Reinforced Adaptive Large Neighborhood Search

DR-ALNS introduces an innovative methodology that leverages Deep Reinforcement Learning (DRL) to configure the Adaptive Large Neighborhood Search (ALNS) algorithm for solving combinatorial optimization problems (COPs). This method learns to select operators, fine-tune parameters, and regulate the acceptance criterion during the search process to dynamically configure ALNS based on the search state, aiming to yield more effective solutions in subsequent iterations. This repository contains the implementation code for DR-ALNS as described in the paper titled "Online Control of Adaptive Large Neighborhood Search Using Deep Reinforcement Learning".

## DR-ALNS Framework

Based on the search status in each iteration, the DRL agent (i.e., a neural network) chooses a destroy and repair operator from the predefined candidates, determines the level of destruction, and adjusts the acceptance criterion parameter (i.e., simulated annealing temperature). These actions are performed in the environment (i.e., ALNS algorithm), which finds a new solution and returns the next state and reward to the agent.

<div style="text-align:center;">
  <img src="https://github.com/RobbertReijnen/DR-ALNS/assets/53526789/5654a71a-3972-4d91-9ce0-86d83faa21d3" alt="DR-ALNS-framework" style="max-width:50%; max-height:50%;">
</div>


## Usage

To use DR-ALNS for solving COPs, follow these steps:

1. Clone the repository: `git clone https://github.com/RobbertReijnen/DR-ALNS.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Configure experiment configuration in the config file (e.g., in 'DR-ALNS/src/routing/cvrp/configs')
4. Run DR-ALNS algorithm: `python run_dr_alns_cvrp.py`

## Citation

If you use DR-ALNS in your research or work, please cite the following paper:

```
@inproceedings{reijnen2024online,
  title={Online control of adaptive large neighborhood search using deep reinforcement learning},
  author={Reijnen, Robbert and Zhang, Yingqian and Lau, Hoong Chuin and Bukhsh, Zaharah},
  booktitle={Proceedings of the International Conference on Automated Planning and Scheduling},
  volume={34},
  pages={475--483},
  year={2024}
}
```
