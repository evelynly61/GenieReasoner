# GenieReasoner

## Unified Embodied VLM Reasoning with Robotic Action via Autoregressive Discretized Pre-training

**Authors**  
Yi Liu<sup>1*</sup>, Sukai Wang<sup>1*†</sup>, Dafeng Wei<sup>1</sup>, Xiaowei Cai<sup>1</sup>, Linqing Zhong<sup>1</sup>, Jiange Yang<sup>1</sup>, Guanghui Ren<sup>2</sup>, Jinyu Zhang<sup>1,3</sup>, Maoqing Yao<sup>2</sup>, Chuankang Li<sup>1</sup>, Xindong He<sup>1</sup>, Liliang Chen<sup>1</sup>, Jianlan Luo<sup>1,3‡</sup>

**Affiliations**  
<sup>1</sup> AgiBot Research  
<sup>2</sup> AgiBot  
<sup>3</sup> Shanghai Innovation Institute

**Links**  
- Paper: https://arxiv.org/abs/2512.24125
- Benchmark: https://github.com/GenieReasoner/ERIQ

## Overview

![Main Teaser Figure](./figs/fig1_vis.png)

While general-purpose robots require both broad semantic reasoning and high-precision execution, existing Vision-Language-Action (VLA) models often struggle with the trade-off between these two critical capabilities. To address this, we present **GenieReasoner**, a unified framework that co-optimizes high-level embodied intelligence and low-level control within a single autoregressive transformer. Our work first establishes **ERIQ**, a comprehensive benchmark designed to quantify the reasoning bottleneck in robotic manipulation. To bridge the gap from reasoning to precise action, we introduce **FACT**, a novel flow-based action tokenizer that preserves high-fidelity trajectories in a discrete space. This unified design allows **GenieReasoner** to significantly outperform both diffusion-based and discrete-action baselines across simulated and real-world tasks.

## Demos

### Embodied Reasoning

- **Logic Reasoning**  
  Identifies the target is blocked and plans a "remove-then-retrieve" action sequence.  
  Video: `./videos/reasoning_inner.mp4`
- **Open-Set Shelf Resetting**  
  Continuously identifies and restores misplaced items, generalizing to arbitrary SKUs.  
  Video: `./videos/reasoning_error.mp4`
- **Semantic Following**  
  Precise semantic alignment for both single-step actions and full task execution.  
  Video: `./videos/instr_cake.mp4`
- **Spatial Understanding**  
  Reasoning over geometric relationships to place objects in distinct relative orientations.  
  Video: `./videos/spatial_blocks.mp4`

### Robust Generalization

- **Water Bottle**  
  UserPrompt: I'm thirsty, but I'm trying to lose weight. Please give me a suitable drink.  
  Videos: `./videos/gen_water.mp4`, `./videos/gen_water_fpv.mp4`
- **Red Pen**  
  UserPrompt: I want to write, please give me the corresponding items.  
  Videos: `./videos/gen_pen.mp4`, `./videos/gen_pen_fpv.mp4`
- **White Wadded Paper**  
  UserPrompt: Pick up desktop trash.  
  Videos: `./videos/gen_paper.mp4`, `./videos/gen_paper_fpv.mp4`
- **Remote Control**  
  UserPrompt: I want to watch TV and switch channels. Please provide me the corresponding items.  
  Videos: `./videos/gen_remote.mp4`, `./videos/gen_remote_fpv.mp4`
- **Charging Cable**  
  UserPrompt: My phone is out of battery, please give me the corresponding item.  
  Videos: `./videos/gen_cable.mp4`, `./videos/gen_cable_fpv.mp4`

### Human-Robot Interaction

- **Cooperation**  
  "Always At Your Service"  
  Video: `./videos/hri_cooperation.mp4`
- **Adversarial**  
  Dynamic Following  
  Video: `./videos/hri_adversarial.mp4`
- **Interaction**  
  Human Intention Understanding & Action Interaction  
  Video: `./videos/hri_coop.mp4`

## Methodology: GenieReasoner

![Framework](./figs/fig_framework.png)

We seek a unified **"Action as Language"** paradigm that enables action sequences to inherit the compositional generalization of large Vision-Language Models (VLMs), while simultaneously preserving the high-precision continuous control required for reliable physical execution.

Most prior VLA systems (e.g., π<sub>0.5</sub>-style) couple a discrete VLM backbone with a continuous policy head to preserve control precision. However, the separation between token-level reasoning and continuous regression can introduce **knowledge insulation**, which may hinder tight reasoning-to-action alignment and lead to weaker generalization in complex, unseen scenarios.

GenieReasoner addresses this via two complementary directions:

1. Unifying perception, reasoning, and action into a single discrete representation to remove cross-objective interference.
2. Introducing **FACT** to discretize continuous trajectories with high-fidelity reconstruction.

This synergistic approach allows the entire model to be optimized with a single autoregressive objective, ensuring that abstract semantic reasoning flows seamlessly into precise physical control without compromise.

## FACT: Bridging Discrete Thought & Continuous Action

FACT encodes continuous actions into discrete tokens to align with the VLM, utilizing Flow Matching to reconstruct high-fidelity trajectories from the quantized representation.

![Tokenizer](./figs/fig_tokenizer.png)

### Components

- **VQ-Encoder**  
  Built on the MM-DiT architecture, it compresses continuous action chunks into **compact discrete tokens**. This transforms complex physical dynamics into a unified vocabulary compatible with the VLM.
- **Flow-Matching Decoder**  
  Leveraging an MM-DiT backbone, it utilizes **Flow Matching** to reconstruct high-fidelity trajectories from discrete tokens. This design ensures smooth, precise motion recovery despite the discrete bottleneck.

### Why It Works Better

![MSE Comparison](./figs/fig_exp_tokenizer.png)

- **High Fidelity**  
  FACT achieves an **order-of-magnitude lower MSE** than FAST+ and significantly smaller tokens than simple binning. Its MM-DiT Flow-Matching decoder eliminates quantization artifacts, reconstructing smooth, continuous trajectories to ensure sub-millimeter precision.
- **Unified Space**  
  Resolves the **gradient interference** that plagues hybrid architectures. By treating actions as discrete tokens, FACT aligns the motor control space with the VLM's reasoning space, allowing both to be co-optimized within a **shared gradient space**.
- **Real-World Success**  
  By leveraging the **FACT action tokenizer** to bridge discrete reasoning and continuous control, GenieReasoner outperforms the discrete baseline (π0-FAST) in instruction following capabilities while surpassing continuous models (e.g., π0.5) in task success rates, ultimately achieving state-of-the-art aggregate performance.

![Language Following](./figs/fig_lang_following.png)
![Task Success](./figs/fig_task_success.png)

## ERIQ: A Large-Scale Benchmark for Embodied Reasoning

### Design Motivation

We posit that advancing **embodied reasoning** is pivotal for achieving generalization and robustness in unstructured environments. Unlike traditional simulation-based VLA evaluations, ERIQ is designed to **decouple and quantify** task-specific reasoning capabilities, measuring the abstract cognition essential for generalization. By establishing this embodied intelligence evaluation suite, we can rigorously assess the abstract reasoning proficiency of pretrained models. This approach facilitates the optimization of multi-stage training and transforms the iterative development of VLAs into a more controllable and guided process.

### Design Principles

ERIQ employs a **standardized Visual Question Answering (VQA) protocol** (multiple-choice or binary) to ensure deterministic, rule-based evaluation, eliminating the ambiguity of open-ended generation metrics.

The framework assesses four pillars of embodied intelligence:

1. Spatial Perception and Grounding
2. Error Detection and Recovery
3. Planning and Monitoring
4. Human-Robot Collaboration

Spanning 15 fine-grained sub-tasks and over 100 scenarios, it tests multi-modal reasoning across static, sequential, and interleaved image-text contexts.

![ERIQ Benchmark Overview](./figs/aer_benchmark_visualization.png)

### Scene Distribution

- Home: 35%
- Restaurant: 20%
- Supermarket: 20%
- Industrial: 15%
- Office: 10%

### Vision Source

- Static: 52.9%
- Sequential: 26.6%
- Interleaved: 20.6%

### 15 Diagnostic Tasks

- Scene Understanding: 967
- Success Detection: 800
- Action Understanding: 600
- Subtask Planning: 600
- Trajectory Understanding: 505
- Task Grounding: 486
- Dualview Matching: 445
- Mistake Existence: 334
- Task Progress: 281
- Relative Pos. Grounding: 249
- Human-Robot Interaction: 227
- Human Intention: 215
- Mistake Classify: 116
- Fine-grained Planning: 112
- Mistake Recovery: 105

## Closing Statement

We have fully open-sourced the **ERIQ Benchmark**, aiming to provide a reproducible and measurable technical foundation for the embodied AI community. We sincerely invite developers and researchers to leverage this benchmark, share feedback on edge cases in real-world scenarios, and collaborate to refine the metrics for embodied reasoning, accelerating the emergence of general-purpose embodied intelligence.

## Citation

```bibtex
@misc{liu2025unifiedembodiedvlmreasoning,
  title={Unified Embodied VLM Reasoning with Robotic Action via Autoregressive Discretized Pre-training},
  author={Yi Liu and Sukai Wang and Dafeng Wei and Xiaowei Cai and Linqing Zhong and Jiange Yang and Guanghui Ren and Jinyu Zhang and Maoqing Yao and Chuankang Li and Xindong He and Liliang Chen and Jianlan Luo},
  year={2025},
  eprint={2512.24125},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2512.24125},
}
```

© 2025 AgiBot Research.
