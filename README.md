# RDACF : A Closed-Loop Decision-Making Framework for Autonomous Driving via Interactive Prediction and Spatiotemporal Risk Propagation
A closed-loop decision-making framework integrating trajectory prediction and adaptive risk-discounting, designed for safe and efficient navigation in mixed traffic scenarios with coexisting Human-Driven Vehicles (HDVs) and Autonomous Vehicles (AVs).
## Core Challenges Addressed
In mixed traffic environments, autonomous decision-making demands accurate multi-step interaction prediction and dynamic risk assessment. Existing methods suffer from three key limitations:
* Neglect of reciprocal influence between ego actions and lack of physical consistency constraints in prediction
* Absence of explicit modeling for spatiotemporal propagation of potential hazards in risk evaluation
* Absence of explicit modeling for spatiotemporal propagation of potential hazards in risk evaluation
## Framework
To address these issues, our framework for the control logic is as follows：


## Result
The results of our framework in the following sample cases：
 1. Turn-Left
    
 2. Merge
    
 3. Cruise
    
 4. Overtake

## How to use
### Create a new Conda environment
```bash
conda create -n smarts python=3.8
```

### Install the SMARTS simulator
```bash
conda activate smarts
```

Install the [SMARTS simulator](https://smarts.readthedocs.io/en/latest/setup.html). 
```bash
# Download SMARTS
git clone https://github.com/huawei-noah/SMARTS.git
cd <path/to/SMARTS>
git checkout comp-1

# Install the system requirements.
bash utils/setup/install_deps.sh

# Install smarts with comp-1 branch.
pip install "smarts[camera-obs] @ git+https://github.com/huawei-noah/SMARTS.git@comp-1"
```

### Install Pytorch
```bash
conda install pytorch==1.12.0 -c pytorch
```

### Training
Run `train.py`. Leave other arguments vacant to use the default setting.
```bash
python train.py --use_exploration --use_interaction
```

### Testing
Run `test.py`. You need specify the path to the trained predictor `--model_path`. You can aslo set `--envision_gui` to visualize the performance of the framework in envision or set `--sumo_gui` to visualize in sumo.
```bash
python test.py --model_path /training_log/Exp/model.pth
```
To visualize in Envision (some bugs exist in showing the road map), you need to manually start the envision server and then go to `http://localhost:8081/`.
```bash
scl envision start -p 8081
```
## Citation
```
@article{huang2023learning,
  title={Learning Interaction-aware Motion Prediction Model for Decision-making in Autonomous Driving},
  author={Huang, Zhiyu and Liu, Haochen and Wu, Jingda and Huang, Wenhui and Lv, Chen},
  journal={arXiv preprint arXiv:2302.03939},
  year={2023}
}
```



