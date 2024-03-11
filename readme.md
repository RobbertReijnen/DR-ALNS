## DR-ALNS: Deep Reinforced Adaptive Large Neighborhood Search

DR-ALNS is a novel approach that leverages Deep Reinforcement Learning (DRL) to configure the Adaptive Large Neighborhood Search (ALNS) algorithm for solving combinatorial optimization problems (COPs). This repository contains the implementation code for DR-ALNS as described in the paper titled "Online Control of Adaptive Large Neighborhood Search Using Deep Reinforcement Learning".

## Overview

The Adaptive Large Neighborhood Search (ALNS) algorithm has shown considerable success in solving combinatorial optimization problems (COPs). However, configuring ALNS with the appropriate selection and acceptance parameters is a complex and resource-intensive task. DR-ALNS addresses this challenge by employing Deep Reinforcement Learning (DRL) to dynamically select operators, adjust parameters, and control the acceptance criterion during the search process.

## Key Features

- **DR-ALNS Framework**: The framework implements the DR-ALNS algorithm, enabling dynamic configuration of ALNS for improved performance in solving COPs.
- **Deep Reinforcement Learning**: Utilizes DRL to learn and adapt ALNS configuration based on the state of the search, resulting in more effective solutions.
- **Evaluation on Orienteering Problem**: The proposed method is evaluated on an orienteering problem with stochastic weights and time windows, showcasing its superiority over existing approaches.
- **Public Implementation Code**: The implementation code for DR-ALNS will be made publicly available to facilitate further research and experimentation.

## Usage

To use DR-ALNS for solving COPs, follow these steps:

1. Clone the repository: `git clone https://github.com/RobbertReijnen/DR-ALNS.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Configure config file parameters: 'e.g., in: DR-ALNS/src/routing/cvrp/configs`
4. Run alns/dr-alns algorithm: `python run_alns_cvrp.py`

## Citation

If you use DR-ALNS in your research or work, please cite the following paper:

