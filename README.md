<h1 align="center"> RAGEN: Training Agents by Reinforcing Reasoning </h1>


<p align="center"><img src="./public/ragen.png" width="800px" alt="RICO Framework" /></p>
<p align="center" style="font-size: 18px;">
  <strong>RAGEN</strong> leverages reinforcement learning to train <strong>LLM reasoning agents</strong> in interactive, stochastic environments.<br>
  <em>We strongly believe in the future of RL + LLM + Agents. The release is a minimally viable leap forward.</em>
</p>


<p align="center">
  <a href="https://ragen-tutorial.readthedocs.io/"><img src="https://img.shields.io/badge/📚_Documentation-4285F4?style=for-the-badge&logoColor=white" alt="Documentation"></a>
  <a href="#"><img src="https://img.shields.io/badge/📝_Blog-FF5722?style=for-the-badge&logoColor=white" alt="Blog"></a>
  <a href="#"><img src="https://img.shields.io/badge/📄_Paper-EA4335?style=for-the-badge&logoColor=white" alt="Paper"></a>
  <a href="#"><img src="https://img.shields.io/badge/🔍_Post-34A853?style=for-the-badge&logoColor=white" alt="Post"></a>
</p>

## Overview

Reinforcement Learning (RL) with rule-based rewards has shown promise in enhancing reasoning capabilities of large language models (LLMs). However, existing approaches have primarily focused on static, single-turn tasks like math reasoning and coding. Extending these methods to agent scenarios introduces two fundamental challenges:

1. **Multi-turn Interactions**: Agents must perform sequential decision-making and react to environment feedback
2. **Stochastic Environments**: Uncertainty where identical actions can lead to different outcomes

RAGEN addresses these challenges through:
- A Markov Decision Process (MDP) formulation for agent tasks
- Reason-Interaction Chain Optimization (RICO) algorithm that optimizes entire trajectory distributions
- Progressive reward normalization strategies to handle diverse, complex environments

## Algorithm

RAGEN introduces a reinforcement learning framework to train reasoning-capable LLM agents that can operate in interactive, stochastic environments. 

<p align="center"><img src="./public/rico.png" width="800px" alt="RICO Framework" /></p>
<p align="center" style="font-size: 16px; max-width: 800px; margin: 0 auto;">
The Reasoning-Interaction Chain Optimization (RICO) framework with two interleaved stages: <b>rollout stage</b> and <b>update stage</b>. LLM iteratively generates reasoning-guided actions to interact with the environment to obtain trajectory-level rewards, normalized for LLM update to jointly optimize reasoning and action strategies.
</p>


<!-- ## Algorithm

RAGEN introduces a reinforcement learning framework to train reasoning-capable LLM agents that can operate in interactive, stochastic environments. The framework consists of two key components:

### > MDP Formulation
We formulate agent-environment interactions as Markov Decision Processes (MDPs) where states and actions are token sequences, allowing LLMs to reason over environment dynamics. At time t, state $s_t$ transitions to state $s_{t+1}$ through action $a_t$ following transition function $\mathcal{P}$: $s_{t+1} \sim \mathcal{P}(\cdot|s_t, a_t)$. The policy $\pi_\theta$ generates actions given the trajectory: $a_t \sim \pi_\theta(\cdot|s_t, [s, a]_{0:t-1})$. The objective is to maximize expected cumulative rewards across multiple interaction turns: $J_{\text{Interactive}}(\theta) = \mathbb{E}_{\substack{s_t \sim \mathcal{D} \\ a_t \sim \pi_{\theta}(\cdot|s_t)}}[\sum_{t} r(s_t,a_t)]$.

### >  Reasoning-Interaction Chain Optimization
RICO enables LLMs to jointly optimize reasoning and action strategies over entire trajectories. The algorithm alternates between two phases:

#### Rollout Stage: Reasoning-Interaction Chain Generation
Given an initial state $s_0$, the LLM generates $N$ trajectories, each with up to $K$ turns. At each step $t$, the model receives the trajectory history $\tau_{1:t-1}$ and generates a reasoning-guided action: $a^T_t = \texttt{<think>...</think><ans>} a_t \texttt{</ans>}$. The environment receives $a_t$ and returns feedback (reward $r_t$ and next state $s_{t+1}$).

#### Update Stage: Multi-turn Trajectory Optimization
After generating trajectories, we train LLMs to optimize expected rewards. Instead of step-by-step optimization, RICO optimizes entire trajectories: $J_{\text{RICO}}(\theta, R) = \mathbb{E}_{\substack{s_0 \sim \mathcal{D} \\ \tau \sim \pi_{\text{old}}(\cdot|s_0)}}\left[\frac{P_\theta(\tau|s_0)}{P_{\text{old}}(\tau|s_0)}R(\tau)\right]$. This approach enables long-horizon reasoning while maintaining computational efficiency.

### > Reward Normalization Strategies
We implement three progressive normalization strategies to stabilize training: (1) **ARPO**: $R^{\text{ARPO}}(r_{\text{all}}^{(i)}) = r_{\text{all}}^{(i)}$ preserves raw rewards; (2) **BRPO**: $R^{\text{BRPO}}(r_{\text{all}}^{(i)}) = (r_{\text{all}}^{(i)} - \mu_B)/\sigma_B$ normalizes across batches; and (3) **GRPO**: $R^{\text{GRPO}}(r_{\text{all}}^{(i)}) = (r_{\text{all}}^{(i)} - \mu_{p_i})/\sigma_{p_i}$ normalizes within prompt groups to balance learning across varying task difficulties. -->

<style>
.math {
  font-family: "Latin Modern Math", "Computer Modern", serif;
  font-style: italic;
}
</style>

<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML"></script>
<script type="text/x-mathjax-config">
MathJax.Hub.Config({
  tex2jax: {
    inlineMath: [['$','$'], ['\\(','\\)']],
    displayMath: [['$$','$$'], ['\\[','\\]']],
    processEscapes: true
  }
});
</script>


<div>
  <p>RAGEN introduces a reinforcement learning framework to train reasoning-capable LLM agents that can operate in interactive, stochastic environments. The framework consists of two key components:</p>

  <h3>> MDP Formulation</h3>
  <p>
    We formulate agent-environment interactions as Markov Decision Processes (MDPs) where states and actions are token sequences, allowing LLMs to reason over environment dynamics. At time t, state <span class="math">s_t</span> transitions to state <span class="math">s_{t+1}</span> through action <span class="math">a_t</span> following transition function <span class="math">\mathcal{P}</span>: <span class="math">s_{t+1} \sim \mathcal{P}(\cdot|s_t, a_t)</span>. The policy <span class="math">\pi_\theta</span> generates actions given the trajectory: <span class="math">a_t \sim \pi_\theta(\cdot|s_t, [s, a]_{0:t-1})</span>. The objective is to maximize expected cumulative rewards across multiple interaction turns: <span class="math">J_{\text{Interactive}}(\theta) = \mathbb{E}_{\substack{s_t \sim \mathcal{D} \\ a_t \sim \pi_{\theta}(\cdot|s_t)}}[\sum_{t} r(s_t,a_t)]</span>.
  </p>

  <h3>> Reasoning-Interaction Chain Optimization</h3>
  <p>
    RICO enables LLMs to jointly optimize reasoning and action strategies over entire trajectories. The algorithm alternates between two phases:
  </p>

  <h4>Rollout Stage: Reasoning-Interaction Chain Generation</h4>
  <p>
    Given an initial state <span class="math">s_0</span>, the LLM generates <span class="math">N</span> trajectories, each with up to <span class="math">K</span> turns. At each step <span class="math">t</span>, the model receives the trajectory history <span class="math">\tau_{1:t-1}</span> and generates a reasoning-guided action: <span class="math">a^T_t = \texttt{<think>...</think><ans>} a_t \texttt{</ans>}</span>. The environment receives <span class="math">a_t</span> and returns feedback (reward <span class="math">r_t</span> and next state <span class="math">s_{t+1}</span>).
  </p>

  <h4>Update Stage: Multi-turn Trajectory Optimization</h4>
  <p>
    After generating trajectories, we train LLMs to optimize expected rewards. Instead of step-by-step optimization, RICO optimizes entire trajectories: <span class="math">J_{\text{RICO}}(\theta, R) = \mathbb{E}_{\substack{s_0 \sim \mathcal{D} \\ \tau \sim \pi_{\text{old}}(\cdot|s_0)}}\left[\frac{P_\theta(\tau|s_0)}{P_{\text{old}}(\tau|s_0)}R(\tau)\right]</span>. This approach enables long-horizon reasoning while maintaining computational efficiency.
  </p>

  <h3>> Reward Normalization Strategies</h3>
  <p>
    We implement three progressive normalization strategies to stabilize training: (1) <strong>ARPO</strong>: <span class="math">R^{\text{ARPO}}(r_{\text{all}}^{(i)}) = r_{\text{all}}^{(i)}</span> preserves raw rewards; (2) <strong>BRPO</strong>: <span class="math">R^{\text{BRPO}}(r_{\text{all}}^{(i)}) = (r_{\text{all}}^{(i)} - \mu_B)/\sigma_B</span> normalizes across batches; and (3) <strong>GRPO</strong>: <span class="math">R^{\text{GRPO}}(r_{\text{all}}^{(i)}) = (r_{\text{all}}^{(i)} - \mu_{p_i})/\sigma_{p_i}</span> normalizes within prompt groups to balance learning across varying task difficulties.
  </p>
</div>


## Performance

We evaluate RAGEN across multiple model sizes and configurations. Below are results from our Sokoban experiments using Qwen-2.5-{0.5B, 3B}-{Instruct, None} and DeepSeek-R1-Distill-Qwen-1.5B.

<img src="./public/loss_curve.png" width="800px" alt="Loss curves for different models" />

Key observations:
- Instruct-finetuned models show early advantages but the gap narrows as training progresses
- Larger models (3B) generally outperform smaller models (0.5B), though the advantage is not dramatic
- The R1-distilled 1.5B model initially underperforms compared to 0.5B models
- Training has not yet converged in these experiments

Our analysis reveals two key aspects of LLM agent training with RL:
1. **Prompt diversity**: Balancing observation variety and effective response comparison
2. **Online rollout frequency**: Mediating between training stability and data recency

## Example Trajectories

Visualization of agent reasoning on the Sokoban task:

<p align="center" style="display: flex; justify-content: center; gap: 10px;">
    <img src="./public/step_1.png" width="200px" alt="Step 1" />
    <img src="./public/step_2.png" width="200px" alt="Step 2" />
</p>

The visualizations show how the agent reasons through sequential steps to solve the puzzle.

## Environment Setup
For detailed setup instructions, please check our [documentation](https://ragen-tutorial.readthedocs.io/). Here's a quick start guide:

```bash
# Setup environment and download data (7MB)
bash scripts/setup_ragen.sh
python scripts/download_data.py
```

If this fails, you can follow the manual setup instructions in `scripts/setup_ragen.md`.

## Training Models
Here's how to train models with RAGEN:

### Create data
We provide 10k first-round-observation data for both Sokoban and FrozenLake tasks.

```bash
# Basic data creation
bash scripts/create_data.sh

# Or for research purposes, create more comprehensive data
bash scripts/create_data_full.sh
```

### Export variables and train
We provide default configuration in `verl/trainer/config/ppo_trainer.yaml`. To train:

```bash
bash train.sh sokoban \
    model.experiment_name=new_test

# Override config parameters as needed
bash train.sh sokoban \
    model.experiment_name=new_test_debug \
    training.train_batch_size=128 \
    training.ppo_batch_size=64
```

## Supervised Finetuning (Optional)
For supervised finetuning with LoRA:

1. Create supervised finetuning data:
```bash
bash sft/generate_data.sh <env_type>
```

2. Finetune the model:
```bash
bash sft/finetune_lora.sh <env_type> <num_gpus> <save_path>
```

3. Merge LoRA weights with the base model:
```bash
python sft/utils/merge_lora.py \
    --base_model_name <base_model_name> \
    --lora_model_path <lora_model_path> \
    --output_path <output_path>
```

## Visualization
To visualize agent trajectories:

1. Set visualization parameters in `train.sh`:
```bash
logging.log_images=True
logging.log_image_dir=log/trajectory
logging.log_image_step_size=4
logging.log_n_image_per_batch=32
```

2. View the visualizations:
```bash
cd log/trajectory
python -m http.server 8000
# Access at http://localhost:8000/[EXP_NAME]/step_[STEP_NUM]/trajectory_data_[ID].html
```

3. For proper font rendering:
```bash
sudo apt-get install fonts-noto-cjk
```

4. Download visualization data from wandb:
```python
from ragen.utils.wandb import download_wandb
download_wandb("RUN_ID") # e.g., 9o465jqj
```

## Case Studies
We provide several case studies showing the model's behavior:
- [Reward hacking](https://github.com/ZihanWang314/agent-r1/blob/main/cases/reward_hacking.txt)
- [Challenging moments](https://github.com/ZihanWang314/agent-r1/blob/main/cases/suck_moment.txt)

More case studies will be added to showcase both successful reasoning patterns and failure modes.

## Feedback
We welcome all forms of feedback! Please raise an issue for bugs, questions, or suggestions. This helps our team address common problems efficiently and builds a more productive community.

## Authors
- [Zihan Wang*](https://zihanwang314.github.io/)
- [Kangrui Wang](https://jameskrw.github.io/)
- [Qineng Wang](https://qinengwang-aiden.github.io/)
- [Pingyue Zhang](https://williamzhangsjtu.github.io/)
- [Manling Li†](https://limanling.github.io)

*: Project Lead; †: Advising.
Remaining authors in alphabetical order.

## Acknowledgements
We thank [DeepSeek](https://github.com/deepseek-ai/DeepSeek-R1) for providing the DeepSeek-R1 model and ideas. We thank the [veRL](https://github.com/volcengine/verl) team for their infrastructure. We thank the [TinyZero](https://github.com/Jiayi-Pan/TinyZero) team for their discoveries that inspired our early exploration. We thank Yiping Lu, Runxin Xu, Kyunghyun Cho for insightful discussions.

## Citation
```md
@misc{RAGEN,
  author       = {Zihan Wang and Kangrui Wang and Qineng Wang and Pingyue Zhang and Manling Li},
  title        = {RAGEN: A General-Purpose Reasoning Agent Training Framework},
  year         = {2025},
  organization = {GitHub},
  url          = {https://github.com/ZihanWang314/ragen},
}
```