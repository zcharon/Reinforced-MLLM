# Reinforced MLLM: A Survey on RL-Based Reasoning in Multimodal Large Language Models

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-GPL3.0-purple.svg)](LICENSE) [![Awesome](https://awesome.re/badge.svg)](https://awesome.re) [![arxiv](https://img.shields.io/badge/arxiv-Paper-red)](https://arxiv.org/abs/2504.21277)

</div>

<div align="center" style="font-family: Arial, sans-serif;">
  <p>
    <a href="#news-" style="text-decoration: none; font-weight: bold;">News 📣</a> - 
    <a href="#methods-" style="text-decoration: none; font-weight: bold;">Methods 📝</a> - 
    <a href="#benchmarks-" style="text-decoration: none; font-weight: bold;">Benchmarks 📈</a>
  </p>
  <p>
    <a href="#contribution-and-acknowledgment-%EF%B8%8F" style="text-decoration: none; font-weight: bold;">Contribution and Acknowledgment ❤️</a> - 
    <a href="#citation-" style="text-decoration: none; font-weight: bold;">Citation 📄</a>
  </p>
</div>

The integration of reinforcement learning (RL) into the reasoning capabilities of Multimodal Large Language Models (MLLMs) has rapidly emerged as a transformative research direction. While MLLMs significantly extend Large Language Models (LLMs) to handle diverse modalities such as vision, audio, and video, enabling robust reasoning across multimodal inputs remains challenging. This survey systematically reviews recent advances in RL-based reasoning for MLLMs, covering key algorithmic designs, reward mechanism innovations, and practical applications. We highlight two main RL paradigms—value-model-free and value-model-based methods—and analyze how RL enhances reasoning abilities by optimizing reasoning trajectories and aligning multimodal information. Furthermore, we provide an extensive overview of benchmark datasets, evaluation protocols, and existing limitations, and propose future research directions to address current bottlenecks such as sparse rewards, inefficient cross-modal reasoning, and real-world deployment constraints. Our goal is to offer a comprehensive and structured guide to researchers interested in advancing RL-based reasoning in the multimodal era.

# News 📣

+ **[2025-04-30] 🔥🔥 We have summarized the MLLM RL-Based Reasoning from January to March 2025 and released the first version of our survey on Reinforced MLLM on [arXiv](https://arxiv.org/abs/2504.21277).**

# Methods 📝

| Model                     | Date | Org | Modality    | Strategy     | Algorithm     | Applications        |
|------------------------|------|-------------|--------------|----------------|----------------------|------------------------|
| [KIMI K1.5](https://arxiv.org/abs/2501.12599) | 1.22    | MoonshotAI | T&I             | SFT&RL             | OPMD                | General Reasoning |
| [MedVLM-R1](https://arxiv.org/abs/2502.19634) | 2.26    | TUM | T&I             | RL                 | GRPO                | Medical                |
| [Visual-RFT](https://arxiv.org/abs/2503.01785) | 3.03    | SJTU | T&I             | RL                 | GRPO                | Detection&CLS          |
| [R1-Omni](https://arxiv.org/abs/2503.05379) | 3.07    | Alibaba | T&V&A           | SFT&RL             | GRPO                | Omni |
| [VisualThinker-R1-Zero](https://arxiv.org/abs/2503.05132) | 3.07    | UCLA | T&I             | RL                 | GRPO                | Spatial Reasoning      |
| [Vision-R1](https://arxiv.org/abs/2503.06749) | 3.09    | ECNU | T&I             | SFT&RL             | GRPO+PTST           | General Reasoning |
| [Seg-Zero](https://arxiv.org/abs/2503.06520) | 3.09    | CUHK | T&I             | RL                 | GRPO                | Segmentation           |
| [GFlowVLM](https://arxiv.org/abs/2503.06514) | 3.09    | HRI-US | T&I             | SFT&RL             | GFlowNet            | General Reasoning |
| [MM-Eureka](https://arxiv.org/abs/2503.07365) | 3.10    | Shanghai AI Lab | T&I             | RL                 | RLOO                | Math                   |
| [Curr-ReFT](https://arxiv.org/abs/2503.07065) | 3.10    | USTC | T&I             | RL                 | GRPO                | General Reasoning |
| [LMM-R1](https://arxiv.org/abs/2503.07536) | 3.10    | SEU | T&I             | RL                 | PPO                 | General Reasoning |
| [R1-Onevision](https://arxiv.org/abs/2503.10615) | 3.13    | ZJU | T&I             | SFT&RL             | GPRO                | General Reasoning |
| [R1-AQA](https://arxiv.org/abs/2503.11197v2) | 3.14    | Xiaomi | T&A             | RL                 | GRPO                | Audio QA               |
| [R1-VL](https://arxiv.org/abs/2503.12937) | 3.17    | NTU | T&I             | SFT&RL             | StepGRPO            | General Reasoning |
| [TimeZero](https://arxiv.org/abs/2503.13377) | 3.17 | RUC | T&V | RL | GRPO | Video Grounding |
| [Skywork R1V](https://arxiv.org/abs/2504.05599) | 3.18    | Skywork AI | T&I | SFT&RL | GRPO | General Reasoning |
| [Med-R1](https://arxiv.org/abs/2503.13939v4) | 3.18 | Emory | T&I | RL | GRPO | Medical |
| [OThink-MR1](https://arxiv.org/abs/2503.16081) | 3.20    | OPPO | T&I             | RL                 | GRPO-D              | General Reasoning |
| [OpenVLThinker](https://arxiv.org/abs/2503.17352) | 3.21    | UCLA | T&I             | SFT&RL             | GRPO                | Math                   |
| [MetaSpatial](https://arxiv.org/abs/2503.18470) | 3.24    | NU  | T&I             | RL                 | GRPO                | Spatial Reasoning      |
| [Reason-RFT](https://arxiv.org/abs/2503.20752) | 3.26    | PKU | T&I             | RL                 | GRPO                | General Reasoning |
| [Video-R1](https://arxiv.org/abs/2504.09641) | 3.27 | CUHK | T&V      | SFT&RL     | T-GRPO       | General Reasoning  |
| [UI-R1](https://arxiv.org/abs/2503.21620) | 3.27 | VIVO | T&I      | RL         | GRPO         | GUI                        |
| [Q-Insight](https://arxiv.org/abs/2503.22679) | 3.28 | PKU | T&I | RL | GRPO | mage Quality |
| [Spatial-R1](https://arxiv.org/abs/2504.01805) | 4.02 | PKU | T&V      | RL         | GRPO         | Spatial Reasoning          |
| [SoTA with Less](https://arxiv.org/abs/2504.07934) | 4.10 | UMD | T&I | RL | GRPO | General Reasoning |
| [Kimi-VL](https://arxiv.org/abs/2504.07491) | 4.10 | MoonshotAI | T&I&V    | SFT&RL     | OPMD         | General Reasoning  |
| [ThinkLite-VL](https://arxiv.org/abs/2504.07934) | 4.10 | UMD | T&I      | RL         | GRPO         | General Reasoning  |
| [Perception-R1](https://arxiv.org/abs/2504.07954) | 4.10 | HUST | T&I      | RL         | GRPO         | General Reasoning  |
| [VideoChat-R1](https://arxiv.org/abs/2504.06958) | 4.10 | Shanghai AI Lab | T&V      | RL         | GRPO         | Spatio-Temporal Perception |
| [VLAA-Thinking](https://arxiv.org/abs/2504.11468) | 4.10 | UCSC | T&I      | RL         | GRPO         | General Reasoning  |
| [VL-Rethinker](https://arxiv.org/abs/2504.08837) | 4.10 | HKUST | T&I      | RL         | GRPO-SSR     | General Reasoning  |
| [R1-Zero-VSI](https://arxiv.org/abs/2504.00883) | 4.14 | SJTU | T&V      | RL         | GRPO         | Spatial Reasoning          |
| [VLM-R1](https://arxiv.org/abs/2504.07615) | 4.14 | ZJU | T&I      | RL         | GRPO         | GUI                        |
| [GUI-R1](https://arxiv.org/abs/2504.10458) | 4.14 | SIAT | T&I      | SFT&RL     | GRPO         | Action Prediction          |
| [TinyLLaVA-Video-R1](https://arxiv.org/abs/2504.09641) | 4.14 | BUAA | T&V      | RL         | GRPO         | Spatio-Temporal Perception |
| [SimpleAR](https://arxiv.org/abs/2504.11455) | 4.15 | FDU | T&I | SFT&RL | GRPO | Image Generation |
| [Embodied-R](https://arxiv.org/abs/2504.12680) | 4.17 | THU | T&V | RL         | GRPO         | Spatial Reasoning          |
| [NoisyRollout](https://arxiv.org/abs/2504.13055) | 4.18 | NUS | T&I      | RL         | GRPO         | Math                       |
| [R1-SGG](https://www.arxiv.org/abs/2504.13617) | 4.18 | HKPU | T&I      | SFT&RL     | GRPO         | Scene Graph Generation     |
| [InfiGUI-R1](https://arxiv.org/abs/2504.14239) | 4.19 | ZJU | T&I      | SFT&RL     | RLOO         | GUI                        |
| [Relation-R1](https://arxiv.org/abs/2504.14642) | 4.20 | HKUST | T&I      | SFT&RL     | GRPO         | Relation Reasoning         |
| [SARI](https://arxiv.org/abs/2504.15900) | 4.22  | Beike | T&A          | SFT&RL                  | GRPO            | Audio Reasoning              |
| [Skywork R1V2](https://arxiv.org/abs/2504.16656) | 4.23 | Skywork AI | T&I      | MPO&RL     | GRPO&SSB     | General Reasoning  |
| [FAST](https://arxiv.org/abs/2504.18458) | 4.25 | ZJU | T&I      | RL         | FAST-GRPO    | General Reasoning  |
| [ChestX-Reasoner](https://arxiv.org/abs/2504.20930) | 4.29  | SJTU | T&I          | SFT&RL                  | GRPO            | Medical                      |
| [T2I-R1](https://arxiv.org/abs/2505.00703) | 5.01 | CUHK | T&I | RL | BiCoT-GRPO | Image Generation |
| [X-Reasoner](https://arxiv.org/abs/2505.03981) | 5.06 | Microsoft | T&I | SFT&RL | GRPO | General Reasoning |
| [EchoInk-R](https://arxiv.org/abs/2505.04623) | 5.07 | CUHK | T&I&V&A | RL | GRPO | Omni |

## Image-Based 🌁

### General Reasoning

**Kimi k1.5: Scaling Reinforcement Learning with LLMs**

| Paper           | [Kimi k1.5: Scaling Reinforcement Learning with LLMs](https://arxiv.org/abs/2501.12599) |
| :-------------- | :----------------------------------------------------------- |
| Link            | [Project🎯](https://github.com/MoonshotAI/Kimi-k1.5)          |
| Reward Function | Rule-Based (Accuracy)                                        |
| Core Insights   | Effective long2short methods that use long-CoT techniques to improve short-CoT models, yielding state-of-the-art short-CoT reasoning results, outperforming existing short-CoT models.The author derives a long-CoT (Chain-of-Thought) RL formulation and employs a variant of online mirror descent to achieve policy optimization. By incorporating sampling strategies, length penalties, and data balancing, the method enables MLLMs (Multi-Layered Learning Models) to acquire chain-of-thought reasoning that exhibits characteristics of planning, reflection, and correction.To alleviate the overthinking phenomenon, the author proposes a length-penalized RL approach and multiple Long2Short methods. |

<details><summary>Show More ...</summary>

**Visual-RFT: Visual Reinforcement Fine-Tuning**

| Paper           | [Visual-RFT: Visual Reinforcement Fine-Tuning](https://arxiv.org/abs/2503.01785) |
| :-------------- | :----------------------------------------------------------- |
| Link            | [Project🎯](https://github.com/Liuziyu77/Visual-RFT)  [Datasets🤗](https://huggingface.co/collections/laolao77/virft-datasets-67bc271b6f2833eccc0651df)  [Code💻](https://github.com/Liuziyu77/Visual-RFT) |
| Reward Function | Rule-based Rewards (Accuracy, IoU, Format)；                 |
| Core Insights   | The R1 training paradigm is applied in the image domain for object detection and classification. |

**Seg-Zero: Reasoning-Chain Guided Segmentation via Cognitive Reinforcement**

| Paper           | [Seg-Zero: Reasoning-Chain Guided Segmentation via Cognitive Reinforcement](https://arxiv.org/abs/2503.06520) |
| :-------------- | :----------------------------------------------------------- |
| Link            | [Datasets🤗](https://huggingface.co/datasets/Ricky06662/refCOCOg_2k_840)   [Models🤗](https://huggingface.co/Ricky06662/Seg-Zero-7B)   [Code💻](https://github.com/dvlab-research/Seg-Zero) |
| Reward Function | Rule-based Rewards (Box IoU, Box L1, Point L1, Format)       |
| Core Insights   | The R1 training paradigm is applied in the field of image segmentation to describe the coordinates of the target to be segmented, followed by invoking a mask model for target segmentation. |

**VisualThinker-R1-Zero: R1-Zero’s “Aha Moment” in Visual Reasoning on a 2B Non-SFT Model**

| Paper           | [R1-Zero’s “Aha Moment” in Visual Reasoning on a 2B Non-SFT Model](https://arxiv.org/abs/2503.05132) |
| :-------------- | :----------------------------------------------------------- |
| Link            | [Code💻](https://github.com/turningpoint-ai/VisualThinker-R1-Zero) |
| Reward Function | Rule-based Rewards (Accuracy-String Match, Format);          |
| Core Insights   | Apply the r1 training paradigm to the image-language domain. |

**Vision-R1: Incentivizing Reasoning Capability in Multimodal Large Language  Models** 

| Paper           | [Vision-R1: Incentivizing Reasoning Capability in Multimodal Large Language  Models](https://arxiv.org/abs/2503.06749) |
| :-------------- | :----------------------------------------------------------- |
| Link            | [Code💻](https://github.com/Osilly/Vision-R1)                 |
| Reward Function | Rule-Based Reward(Accuracy, Format)                          |
| Core Insights   | The author utilized MLLM and DeepSeek-R1 to construct a high-quality 200K multimodal Chain-of-Thought (CoT), which requires no human annotation and serves as the cold-start initialization data for MLLMs.The author proposed the PTST method, combined with GRPO-based RLVR, effectively addressing the "overthinking" optimization issue during reinforcement learning training. |

**MM-Eureka: Exploring Visual Aha Moment with Rule-based Large-scale Reinforcement Learning**

| Paper           | [MM-Eureka: Exploring Visual Aha Moment with Rule-based Large-scale Reinforcement Learning](https://arxiv.org/abs/2503.07365) |
| :-------------- | :----------------------------------------------------------- |
| Link            | [Datasets🤗](https://huggingface.co/datasets/FanqingM/MM-Eureka-Dataset)   [Models🤗](https://huggingface.co/FanqingM)   [Code💻](https://github.com/ModalMinds/MM-EUREKA) |
| Reward Function | Rule-based Rewards (Accuracy, Format)                        |
| Core Insights   | Apply the r1 training paradigm to the image-text domain.     |

**Curr-ReFT: Boosting the Generalization and Reasoning of Vision Language Models with  Curriculum Reinforcement Learning**

| Paper           | [Boosting the Generalization and Reasoning of Vision Language Models with  Curriculum Reinforcement Learning](https://arxiv.org/abs/2503.07065) |
| :-------------- | :----------------------------------------------------------- |
| Link            | [Datasets🤗](https://huggingface.co/datasets/ZTE-AIM/Curr-ReFT-data)   [Models🤗](https://huggingface.co/ZTE-AIM)   [Code💻](https://github.com/ding523/Curr_REFT) |
| Reward Function | Difficulty-aware Rule-based Rewards (Accuracy, IoU, Format)  |
| Core Insights   | The authors propose Curr-ReFT, a novel post-training paradigm that combines curriculum reinforcement learning with reject-sampling based selfimprovement. |

**LMM-R1: Empowering 3B LMMs with Strong Reasoning Abilities Through  Two-Stage Rule-Based RL**

| Paper           | [LMM-R1: Empowering 3B LMMs with Strong Reasoning Abilities Through  Two-Stage Rule-Based RL](https://arxiv.org/abs/2503.07536) |
| :-------------- | :----------------------------------------------------------- |
| Link            | [Code💻](https://github.com/TideDra/lmm-r1)                   |
| Reward Function | Rule-Based Reward (Accuracy, Format)                         |
| Core Insights   | The R1 training paradigm is applied to the image-Text domain. |

**R1-Onevision: Advancing Generalized Multimodal Reasoning through Cross-Modal Formalization** 

| Paper           | [R1-Onevision: Advancing Generalized Multimodal Reasoning through Cross-Modal Formalization](https://arxiv.org/abs/2503.10615) |
| :-------------- | :----------------------------------------------------------- |
| Link            | [Datasets🤗](https://huggingface.co/datasets/Fancy-MLLM/R1-Onevision)   [Models🤗](https://huggingface.co/Fancy-MLLM/R1-Onevision-7B)   [Code💻](https://github.com/Fancy-MLLM/R1-Onevision) |
| Reward Function | Rule-Based Reward (Accuracy, Format)                         |
| Core Insights   | The authors propose a cross-modal reasoning pipeline that transforms images into formal textural representations, enabling precise language-based reasoning.The authors introduce R1Onevision-Bench, a benchmark aligned with human educational stages, covering exams from junior high school to university and beyond. |

**R1-VL: Learning to Reason with Multimodal Large Language Models via Step-wise Group Relative Policy Optimization**

| Paper           | [R1-VL: Learning to Reason with Multimodal Large Language Models via Step-wise Group Relative Policy Optimization](https://arxiv.org/abs/2503.12937) |
| :-------------- | :----------------------------------------------------------- |
| Link            | [Models🤗](https://huggingface.co/jingyiZ00)   [Code💻](https://github.com/jingyi0000/R1-VL) |
| Reward Function | StepRAR（Step-wise Reasoning Accuracy Reward, StepRVR（Step-wise Reasoning Validity Reward） |
| Core Insights   | Introduces step-wise reward mechanisms to address sparse feedback, enabling structurally consistent and self-improving reasoning in MLLMs. |

**Skywork R1V: Pioneering Multimodal Reasoning with Chain-of-Thought**

| Paper           | [Skywork R1V: Pioneering Multimodal Reasoning with Chain-of-Thought](https://arxiv.org/abs/2504.05599) |
| :-------------- | :----------------------------------------------------------- |
| Link            | [Models🤗](https://huggingface.co/Skywork/Skywork-R1V-38B)   [Code💻](https://github.com/SkyworkAI/Skywork-R1V) |
| Reward Function | Rule-based reward (Accuracy, Format)                         |
| Core Insights   | The insight behind the method lies in decoupling the alignment of visual-language representations from the preservation of reasoning capabilities. The authors propose to align the vision encoder with the reasoning-capable language backbone by solely training an MLP-based vision adapter. This approach enables the model to acquire visual perception capabilities while retaining its reasoning abilities. The authors propose a hybrid optimization strategy that combines Iterative Supervised Fine-Tuning (SFT) with Group Relative Policy Optimization (GRPO), significantly enhancing cross-modal integration efficiency. Additionally, they introduce an adaptive-length Chain-of-Thought distillation approach for reasoning data generation. This approach dynamically optimizes reasoning chain lengths, thereby enhancing inference efficiency and preventing excessive reasoning overthinking. |

**OThink-MR1: Stimulating Multimodal Generalized Reasoning Capabilities via Dynamic Reinforcement Learning**

| Paper           | [OThink-MR1: Stimulating Multimodal Generalized Reasoning Capabilities via Dynamic Reinforcement Learning](https://arxiv.org/abs/2503.16081) |
| :-------------- | :----------------------------------------------------------- |
| Link            | -                                                            |
| Reward Function | Rule-Based Reward(Accuracy, Format)                          |
| Core Insights   | Introduces GRPO-D, a dynamic-KL RL strategy that significantly improves cross-task generalization for multimodal reasoning. |

**OpenVLThinker: An Early Exploration to Complex Vision-Language Reasoning via Iterative Self-Improvement** 

| Paper           | [OpenVLThinker: An Early Exploration to Complex Vision-Language Reasoning via Iterative Self-Improvement](https://arxiv.org/abs/2503.17352) |
| :-------------- | :----------------------------------------------------------- |
| Link            | [Models🤗](https://huggingface.co/ydeng9/OpenVLThinker-7B)   [Code💻](https://github.com/yihedeng9/OpenVLThinker) |
| Reward Function | Relu-Based Reward(Format);                                   |
| Core Insights   | Combines iterative SFT and GRPO to endow LVLMs with structured and verifiable multi-step reasoning, achieving strong generalization across visual tasks. |

**Reason-RFT: Reinforcement Fine-Tuning for Visual Reasoning**

| Paper           | [Reason-RFT: Reinforcement Fine-Tuning for Visual Reasoning](https://arxiv.org/abs/2503.20752) |
| :-------------- | :----------------------------------------------------------- |
| Link            | [Project🎯](https://tanhuajie.github.io/ReasonRFT/)  [Datasets🤗](https://huggingface.co/datasets/tanhuajie2001/Reason-RFT-CoT-Dataset)  [Code💻](https://github.com/tanhuajie/Reason-RFT) |
| Reward Function | Rule-Based Reward (Accuracy, Format)                         |
| Core Insights   | The R1 training paradigm is applied to the audio question-answering domain.The author categorizes mathematical reasoning into three types: Discrete-valued Type, Mathematical Type, and Function-based Type, and designs corresponding accuracy reward functions for each type. |

**SoTA with Less: MCTS-Guided Sample Selection for Data-Efficient Visual Reasoning Self-Improvement** 

| Paper           | [SoTA with Less: MCTS-Guided Sample Selection for Data-Efficient Visual Reasoning Self-Improvement](https://arxiv.org/abs/2504.07934) |
| :-------------- | :----------------------------------------------------------- |
| Link            | [Datasets🤗](https://huggingface.co/collections/russwang/thinklite-vl-67f88c6493f8a7601e73fe5a)   [Models🤗](https://huggingface.co/russwang/ThinkLite-VL-7B)   [Code💻](https://github.com/si0wang/ThinkLite-VL) |
| Reward Function | Rule-Based Reward (Accuracy + Format)                        |
| Core Insights   | The author introduces an MCTS-based selection method that quantifies sample difficulty according to the number of iterations required by VLMs to solve each problem. Ultimately, 11k samples are filtered out from a pool of 70k open-source training samples, and ThinkLite-VL is obtained through RFT. |

**Perception-R1: Pioneering Perception Policy with Reinforcement Learning**

| Paper           | [Perception-R1: Pioneering Perception Policy with Reinforcement Learning](https://arxiv.org/abs/2504.07954) |
| :-------------- | :----------------------------------------------------------- |
| Link            | [Datasets🤗](https://huggingface.co/collections/Kangheng/perception-r1-67f6b14f89d307a0ece985af)   [Models🤗](https://huggingface.co/collections/Kangheng/perception-r1-67f6b14f89d307a0ece985af)   [Code💻](https://github.com/linkangheng/PR1) |
| Reward Function | Rule-Based Reward (Accuracy/IoU/Detects&Count/Euclidean Distance/F1 score + Format) |
| Core Insights   | The authors propose Perception-R1, a scalable RL framework using GRPO during MLLM post-training and explore the effects of various RL on different perception tasks. |

**VLAA-Thinker: SFT or RL? An Early Investigation into Training R1-Like Reasoning Large Vision-Language Models**

| Paper           | [SFT or RL? An Early Investigation into Training R1-Like Reasoning Large Vision-Language Models](https://arxiv.org/abs/2504.11468) |
| :-------------- | :----------------------------------------------------------- |
| Link            | [Datasets🤗](https://huggingface.co/collections/UCSC-VLAA/vlaa-thinker-67eda033419273423d77249e)   [Models🤗](https://huggingface.co/collections/UCSC-VLAA/vlaa-thinker-67eda033419273423d77249e)   [Code💻](https://huggingface.co/collections/UCSC-VLAA/vlaa-thinker-67eda033419273423d77249e) |
| Reward Function | Rule-Based Reward, Open-ended Reward                         |
| Core Insights   | The authors found that while Supervised Fine-Tuning (SFT) helps models learn reasoning formats, it tends to lock aligned models into imitative, rigid reasoning patterns, thereby hindering further learning. To systematically study this effect, the authors introduced a multimodal dataset called VLAA-Thinking. Additionally, building on GRPO, they employed a novel hybrid reward module that integrates perceptual and cognitive signals to encourage more authentic and adaptable reasoning behavior. |

**VL-Rethinker: Incentivizing Self-Reflection of Vision-Language Models with Reinforcement Learning**

| Paper           | [VL-Rethinker: Incentivizing Self-Reflection of Vision-Language Models with Reinforcement Learning](https://arxiv.org/abs/2504.08837) |
| :-------------- | :----------------------------------------------------------- |
| Link            | [Datasets🤗](https://huggingface.co/datasets/TIGER-Lab/ViRL39K)   [Models🤗](https://huggingface.co/collections/TIGER-Lab/vl-rethinker-67fdc54de07c90e9c6c69d09)   [Code💻](https://github.com/TIGER-AI-Lab/VL-Rethinker) |
| Reward Function | Rule-Based Reward (Accuracy, Format)                         |
| Core Insights   | The authors adapt the GRPO algorithm with a novel technique called Selective Sample Replay (SSR) to address the vanishing advantages problem. To further encourage slow-thinking, they introduce Forced Rethinking, which appends a textual rethinking trigger to the end of initial rollouts in RL training, explicitly enforcing a self-reflection reasoning step. |

**VLM-R1: A Stable and Generalizable R1-style Large Vision-Language Model** 

| Paper           | [VLM-R1: A Stable and Generalizable R1-style Large Vision-Language Model](https://arxiv.org/abs/2504.07615) |
| :-------------- | :----------------------------------------------------------- |
| Link            | [Datasets🤗](https://huggingface.co/datasets/omlab/VLM-R1)   [Models🤗](https://huggingface.co/omlab/Qwen2.5VL-3B-VLM-R1-REC-500steps)   [Code💻](https://github.com/om-ai-lab/VLM-R1) |
| Reward Function | Rule-Based Reward (Format, Reward)                           |
| Core Insights   | The authors develop VLM-R1, a dedicated framework designed to harness RL for improving VLMs’ performance on general vision-language tasks and select two visual understanding tasks — Referring Expression Compression (REC) and Open-Vocabulary Object Detection (OVD) — to explore the feasibility and effectiveness of applying RL to VLMs. |

**Relation-R1: Cognitive Chain-of-Thought Guided Reinforcement Learning for Unified Relational Comprehension** 

| Paper           | [Relation-R1: Cognitive Chain-of-Thought Guided Reinforcement Learning for Unified Relational Comprehension](https://arxiv.org/abs/2504.14642) |
| :-------------- | :----------------------------------------------------------- |
| Link            | [Code💻](https://github.com/HKUST-LongGroup/Relation-R1)      |
| Reward Function | Rule-Based Reward (Accuracy, Format)                         |
| Core Insights   | The authors propose a unified relation comprehension framework Relation-R1, that integrates Supervised Fine-Tuning (SFT) and RL to empower MLLMs with relational reasoning and generalization capability. |

**Skywork R1V2: Multimodal Hybrid Reinforcement Learning for Reasoning**

| Paper           | [Skywork R1V2: Multimodal Hybrid Reinforcement Learning for Reasoning](https://arxiv.org/abs/2504.16656) |
| :-------------- | :----------------------------------------------------------- |
| Link            | [Models🤗](https://huggingface.co/collections/Skywork/skywork-r1v2-68075a3d947a5ae160272671)   [Code💻](https://github.com/SkyworkAI/Skywork-R1V) |
| Reward Function | Reward Model + Rule-Based Reward (Accuracy, Format)          |
| Core Insights   | R1V2 introduces a hybrid RL paradigm that jointly leverages the Mixed Preference Optimization (MPO) and the GRPO, which harmonizes reward-model guidance with rule-based strategies. The authors introduce the Selective Sample Buffer (SSB) mechanism, which effectively counters the “Vanishing Advantages” dilemma inherent in GRPO. |

**FAST: Fast-Slow Thinking for Large Vision-Language Model Reasoning**

| Paper           | [Fast-Slow Thinking for Large Vision-Language Model Reasoning](https://arxiv.org/abs/2504.18458) |
| :-------------- | :----------------------------------------------------------- |
| Link            | [Code💻](https://github.com/Mr-Loevan/FAST)                   |
| Reward Function | Rule-Based Reward (Accuracy, Format, Thinking)               |
| Core Insights   | The authors present FAST1, a novel Fast-Slow Thinking framework that dynamically adapts reasoning depth based on question characteristics and develop FASTGRPO with three components: model-based metrics for question characterization, an adaptive thinking reward mechanism, and difficulty-aware KL regularization. |

**NoisyRollout: Reinforcing Visual Reasoning with Data Augmentation**

| Paper           | [NoisyRollout: Reinforcing Visual Reasoning with Data Augmentation](https://arxiv.org/abs/2504.13055) |
| :-------------- | :----------------------------------------------------------- |
| Link            | [Datasets🤗](https://huggingface.co/collections/xyliu6/noisyrollout-67ff992d1cf251087fe021a2)   [Models🤗](https://huggingface.co/collections/xyliu6/noisyrollout-67ff992d1cf251087fe021a2)   [Code💻](https://github.com/John-AI-Lab/NoisyRollout) |
| Reward Function | Rule-based Binary Reward                                     |
| Core Insights   | Introduces a hybrid rollout strategy mixing clean and noisy visual inputs during GRPO training to enhance exploration and generalization in VLMs, achieving strong OOD reasoning with minimal data. |

**R1-SGG: Compile  Scene Graphs with Reinforcement Learning**

| Paper           | [R1-SGG: Compile Scene Graphs with Reinforcement Learning](https://www.arxiv.org/abs/2504.13617) |
| :-------------- | :----------------------------------------------------------- |
| Link            | [Code💻](https://github.com/gpt4vision/R1-SGG)                |
| Reward Function | Format Reward, Node-Level Reward, Edge-Level Reward          |
| Core Insights   | Introduces a graph-centric RL framework for multimodal LLMs to generate accurate and structured scene graphs using rule-based rewards on both node and edge levels, drastically reducing generation failures and improving relationship reasoning. |

**SimpleAR: Pushing the Frontier of Autoregressive Visual Generation through Pretraining, SFT, and RL**

| Paper           | [SimpleAR: Pushing the Frontier of Autoregressive Visual Generation through Pretraining, SFT, and RL](https://arxiv.org/abs/2504.11455) |
| :-------------- | :----------------------------------------------------------- |
| Link            | [Models🤗](https://huggingface.co/collections/Daniel0724/simplear-6805053f5b4b9961ac025136)   [Code💻](https://github.com/wdrink/SimpleAR) |
| Reward Function | CLIP-ViT-H-14 Reward, HPSv2 Reward                           |
| Core Insights   | Demonstrates that a vanilla 0.5B autoregressive model trained with SFT and GRPO can rival diffusion models on text-to-image benchmarks through reward-aligned generation and inference acceleration. |

**T2I-R1: Reinforcing Image Generation with Collaborative Semantic-level and Token-level CoT**

| Paper           | [T2I-R1: Reinforcing Image Generation with Collaborative Semantic-level and Token-level CoT](https://arxiv.org/abs/2505.00703) |
| :-------------- | :----------------------------------------------------------- |
| Link            | [Code💻](https://github.com/CaraJ7/T2I-R1)                    |
| Reward Function | Ensemble of vision experts (Human Preference Model, Object Detector, VQA Model, Output Reward Model) |
| Core Insights   | The authors propose a novel reasoning-enhanced text-to-image model powered by reinforcement learning with a bi-level chain-of-thought (CoT) reasoning process. Semantic-level CoT focuses on high-level planning of the image generation, while token-level CoT handles low-level pixel processing during patch-by-patch generation. By optimizing both levels of CoT through an ensemble of reward models, the approach achieves significant performance improvements on complex prompts and uncommon scenarios. |

| Paper           | [X-REASONER: A Simple Yet Effective Post-Training Recipe for Generalizable Reasoning](http://arxiv.org/abs/2412.18925) |
| :-------------- | :----------------------------------------------------------- |
| Link            | [Code💻](https://github.com/microsoft/x-reasoner)             |
| Reward Function | Verifiable rewards based on mathematical textual questions   |
| Core Insights   | The authors introduce X-REASONER, a post-training recipe that enhances the reasoning capabilities of vision-language models using general-domain text-based supervision through combined SFT and RL strategies. X-REASONER demonstrates strong generalization across modalities and domains, achieving state-of-the-art performance on various benchmarks without multimodal training data. Additionally, the authors propose X-REASONER-MED, a medical-specialized variant that achieves new state-of-the-art results on medical tasks. |

</details>

### Vertical Domain

**MetaSpatial: Reinforcing 3D Spatial Reasoning in VLMs for the Metaverse [Metaverse]**

| Paper           | [MetaSpatial: Reinforcing 3D Spatial Reasoning in VLMs for the Metaverse](https://arxiv.org/abs/2503.18470) |
| :-------------- | :----------------------------------------------------------- |
| Link            | [Datasets🤗](https://huggingface.co/datasets/zhenyupan/3d_layout_reasoning)    [Code💻](https://github.com/PzySeere/MetaSpatial) |
| Reward Function | Format Reward, Physics Reward, Rendering-based Reward        |
| Core Insights   | Enables VLMs to learn adaptive 3D spatial reasoning via multi-turn refinement and rule-based rewards, without requiring ground-truth layouts. |

<details><summary>Show More ...</summary>

**UI-R1: Enhancing Action Prediction of GUI Agents by Reinforcement Learning  [GUI]**

| Paper           | [UI-R1: Enhancing Action Prediction of GUI Agents by Reinforcement Learning](https://arxiv.org/abs/2503.21620) |
| :-------------- | :----------------------------------------------------------- |
| Link            | -                                                            |
| Reward Function | Rule-Based Reward (Action, Coordinate, Format)               |
| Core Insights   | Introduces rule-based RL to GUI action prediction with structured rewards, enabling data-efficient generalization across platforms. |

**GUI-R1 : A Generalist R1-Style Vision-Language Action Model For GUI Agents  [GUI]** 

| Paper           | [GUI-R1 : A Generalist R1-Style Vision-Language Action Model For GUI Agents](https://arxiv.org/abs/2504.10458) |
| :-------------- | :----------------------------------------------------------- |
| Link            | [Datasets🤗](https://huggingface.co/datasets/ritzzai/GUI-R1)   [Models🤗](https://huggingface.co/ritzzai/GUI-R1)   [Code💻](https://github.com/ritzz-ai/GUI-R1) |
| Reward Function | Relu-Based Reward(Format, Accuracy, Action, Click, Input, Response) |
| Core Insights   | The authors propose GUI-R1, the first reinforcement learning framework designed to enhance the GUI capabilities of LVLMs in high-level real-world task scenarios, through unified action space rule modeling. |

**InfiGUI-R1: Advancing Multimodal GUI Agents from Reactive Actors to Deliberative Reasoners  [GUI]**

| Paper           | [InfiGUI-R1: Advancing Multimodal GUI Agents from Reactive Actors to Deliberative Reasoners](https://arxiv.org/abs/2504.14239) |
| :-------------- | :----------------------------------------------------------- |
| Link            | [Models🤗](https://huggingface.co/Reallm-Labs/InfiGUI-R1-3B)   [Code💻](https://github.com/Reallm-Labs/InfiGUI-R1) |
| Reward Function | Format Reward, Accuracy Reward, Sub-goal Guidance Reward,  Recovery Reward |
| Core Insights   | Proposes Actor2Reasoner, a two-stage training framework combining spatial reasoning distillation and reinforcement-based deliberation enhancement, effectively evolving GUI agents from reactive to deliberative reasoners. |

**Q-Insight: Understanding Image Quality via Visual Reinforcement Learning  [Image Quality]**

| Paper           | [Q-Insight: Understanding Image Quality via Visual Reinforcement Learning](https://arxiv.org/abs/2503.22679) |
| :-------------- | :----------------------------------------------------------- |
| Link            | [Code💻](https://github.com/lwq20020127/Q-Insight)            |
| Reward Function | Format Reward, Score Regression, Degradation Perception      |
| Core Insights   | Q-Insight unifies score regression and degradation perception via GRPO, enabling interpretable and generalizable image quality understanding. |

**Embodied-R: Collaborative Framework for Activating Embodied Spatial Reasoning in Foundation Models via Reinforcement Learning [Embodied]**

| Paper           | [Embodied-R: Collaborative Framework for Activating Embodied Spatial Reasoning in Foundation Models via Reinforcement Learning](https://arxiv.org/abs/2504.12680) |
| :-------------- | :----------------------------------------------------------- |
| Link            | [Code💻](https://github.com/EmbodiedCity/Embodied-R.code)     |
| Reward Function | Format Reward, Accuracy Reward, Logical Consistency Reward   |
| Core Insights   | Embodied-R decouples perception and reasoning by collaborating on a large VLM and a small LM, and uses reinforcement learning with logical consistency rewards to activate slow-thinking spatial reasoning under limited computational resources. |

**MedVLM-R1: Incentivizing Medical Reasoning Capability of Vision-Language Models via Reinforcement Learning [Medical]** 

| Paper           | [MedVLM-R1: Incentivizing Medical Reasoning Capability of Vision-Language Models via Reinforcement Learning](https://arxiv.org/abs/2502.19634) |
| :-------------- | :----------------------------------------------------------- |
| Link            | [Models🤗](https://huggingface.co/JZPeterPan/MedVLM-R1)       |
| Reward Function | Rule-Based Reward (Accuracy, Format)                         |
| Core Insights   | Leverages GRPO to enable explicit reasoning in medical VLMs without CoT supervision, achieving strong OOD generalization. |

**Med-R1: Reinforcement Learning for Generalizable Medical Reasoning in Vision-Language Models**

| Paper           | [Med-R1: Reinforcement Learning for Generalizable Medical Reasoning in Vision-Language Models](https://arxiv.org/abs/2503.13939) |
| :-------------- | :----------------------------------------------------------- |
| Link            | [Code💻](https://huggingface.co/yuxianglai117/Med-R1)   [Models🤗](https://github.com/Yuxiang-Lai117/Med-R1) |
| Reward Function | Rule-Based Reward (Accuracy, Format)                         |
| Core Insights   | The authors propose Med-R1, which uses reinforcement learning to improve generalization and reliability in medical reasoning across diverse imaging modalities. Med-R1 achieves a 29.94% improvement in average accuracy over its base model and outperforms much larger models like Qwen2-VL-72B. The study also challenges the assumption that more reasoning always helps, showing that omitting intermediate rationales can lead to better generalization with less training. |

**ChestX-Reasoner: Advancing Radiology Foundation Models with Reasoning through Step-by-Step Verification  [Medical]**

| Paper           | [ChestX-Reasoner: Advancing Radiology Foundation Models with Reasoning through Step-by-Step Verification](https://arxiv.org/abs/2504.20930) |
| :-------------- | :----------------------------------------------------------- |
| Link            | -                                                            |
| Reward Function | Format Reward, Outcome Accuracy Reward, Process Factuality Reward |
| Core Insights   | ChestX-Reasoner proposes to mine "step-by-step reasoning supervision signals" from real clinical reports, construct structured reasoning data, and introduce process rewards to optimize the medical diagnostic reasoning capabilities of MLLMs. |

</details>

## Video-Based 📹

| Paper           | [TimeZero: Temporal Video Grounding with Reasoning-Guided LVLM](https://arxiv.org/abs/2503.13377) |
| :-------------- | :----------------------------------------------------------- |
| Link            | [Datasets🤗](https://huggingface.co/datasets/Video-R1/Video-R1-data)   [Models🤗](https://huggingface.co/Video-R1/Video-R1-7B)   [Code💻](https://github.com/tulerfeng/Video-R1) |
| Reward Function | Rule-Based Reward (Template Reward, IoU Reward)              |
| Core Insights   | The authors propose TimeZero, a reasoning-driven LVLM for the temporal video grounding (TVG) task. It uses reinforcement learning to enhance video-language relationship reasoning before making predictions. TimeZero achieves state-of-the-art performance on Charades-STA and demonstrates strong generalization capabilities on out-of-domain tests. The incorporation of Chain-of-Thought (CoT) during training and inference further improves its performance. |

<details><summary>Show More ...</summary>

**Video-R1: Reinforcing Video Reasoning in MLLMs** 

| Paper           | [Video-R1: Reinforcing Video Reasoning in MLLMs](https://arxiv.org/abs/2503.21776) |
| :-------------- | :----------------------------------------------------------- |
| Link            | [Models🤗](https://huggingface.co/wwwyyy/TimeZero-Charades-7B)   [Code💻](https://github.com/www-Ye/TimeZero) |
| Reward Function | Rule-Based Reward (Accuracy, Format, Length Penalty)         |
| Core Insights   | The authors first propose the T-GRPO algorithm, which encourages models to utilize temporal information in videos for reasoning.The authors incorporate high-quality image-reasoning data into the training process: Video-R1-COT-165k for SFT cold start and Video-R1-260k for RL training, both comprising image and video data. |

**R1-Zero-VSI: Improved Visual-Spatial Reasoning via R1-Zero-Like Training**

| Paper           | [Improved Visual-Spatial Reasoning via R1-Zero-Like Training](https://arxiv.org/abs/2504.00883) |
| :-------------- | :----------------------------------------------------------- |
| Link            | [Code💻](https://github.com/zhijie-group/R1-Zero-VSI)         |
| Reward Function | Rule-Based Reward (Format, Reward)                           |
| Core Insights   | Apply the R1 training paradigm to evaluate the ability of small-sized models in video spatial reasoning, and three different think templates were tested. |

**TinyLLaVA-Video-R1: Towards Smaller LMMs for Video Reasoning**

| Paper           | [TinyLLaVA-Video-R1: Towards Smaller LMMs for Video Reasoning](https://arxiv.org/abs/2504.09641) |
| :-------------- | :----------------------------------------------------------- |
| Link            | [Models🤗](https://huggingface.co/Zhang199/TinyLLaVA-Video-R1)   [Code💻](https://github.com/ZhangXJ199/TinyLLaVA-Video-R1) |
| Reward Function | Format Reward, Accuracy Reward, Final Reward Rule            |
| Core Insights   | Demonstrates that even 3B-scale video-language models can achieve strong reasoning performance on general Video-QA benchmarks when guided with structured GRPO reinforcement learning and length-sensitive format rewards. |

**VideoChat-R1: Enhancing Spatio-Temporal Perception via Reinforcement Fine-Tuning**

| Paper           | [VideoChat-R1: Enhancing Spatio-Temporal Perception via Reinforcement Fine-Tuning](https://arxiv.org/abs/2504.06958) |
| :-------------- | :----------------------------------------------------------- |
| Link            | [Models🤗](https://huggingface.co/collections/OpenGVLab/videochat-r1-67fbe26e4eb08c83aa24643e)   [Code💻](https://github.com/OpenGVLab/VideoChat-R1) |
| Reward Function | Format Reward, IoU Reward, Accuracy Reward, Recall Reward (Caption) |
| Core Insights   | Demonstrates that GRPO-based reinforcement fine-tuning on a small amount of spatio-temporal data can significantly enhance video MLLM perception and reasoning without sacrificing general chat ability, outperforming supervised fine-tuning and baseline models. |

**Spatial-R1: Enhancing MLLMs in Video Spatial Reasoning**

| Paper           | [Spatial-R1: Enhancing MLLMs in Video Spatial Reasoning](https://arxiv.org/abs/2504.01805) |
| :-------------- | :----------------------------------------------------------- |
| Link            | [Datasets🤗](https://github.com/OuyangKun10/Spatial-R1/blob/main/annotation/SR-91k.jsonl)   [Models🤗](https://huggingface.co/RUBBISHLIKE/Sptial-R1-exp-1500)   [Code💻](https://github.com/OuyangKun10/Spatial-R1) |
| Reward Function | Format Reward, Numerical Accuracy Reward, Multiple Choice Accuracy Reward |
| Core Insights   | Spatial-R1 proposes a task-aware GRPO reinforcement learning strategy, which, combined with the high-quality automated spatial reasoning dataset SR, significantly improves the ability of MLLMs to perform complex reasoning such as spatial scale, direction, and sequence in videos, surpassing GPT-4o and multiple open source models on VSI-Bench. |

</details>

## Audio-Based 🎧

**R1-AQA: Reinforcement Learning Outperforms Supervised Fine-Tuning: A Case Study on Audio Question Answering**

| Paper           | [Reinforcement Learning Outperforms Supervised Fine-Tuning: A Case Study on Audio Question Answering](https://arxiv.org/abs/2503.11197) |
| :-------------- | :----------------------------------------------------------- |
| Link            | [Models🤗](https://huggingface.co/mispeech/r1-aqa)   [Code💻](https://github.com/xiaomi-research/r1-aqa) |
| Reward Function | Rule-Based Reward (Accuracy, Format)                         |
| Core Insights   | Applies GRPO to LALMs under limited supervision, achieving strong gains on AQA tasks with minimal data. |

<details><summary>Show More...</summary>

**SARI: Structured Audio Reasoning via Curriculum-Guided Reinforcement Learning**

| Paper           | [SARI: Structured Audio Reasoning via Curriculum-Guided Reinforcement Learning](https://arxiv.org/abs/2504.15900) |
| :-------------- | :----------------------------------------------------------- |
| Link            | -                                                            |
| Reward Function | Format Reward, Curriculum-based Reward Scheduling)           |
| Core Insights   | SARI proposes a structured Chain-of-Thought (CoT) + curriculum-guided GRPO fine-tuning framework, systematically evaluating the benefits of structured and unstructured reasoning strategies for audio reasoning models. The final model achieves a 67.08% SOTA performance on MMAU and demonstrates strong cross-domain generalization capabilities on MMSU. |

## Omni 🤖

**R1-Omni: Explainable Omni-Multimodal Emotion Recognition with Reinforcement Learning**

| Paper           | [R1-Omni: Explainable Omni-Multimodal Emotion Recognition with Reinforcement Learning](https://arxiv.org/abs/2503.05379) |
| :-------------- | :----------------------------------------------------------- |
| Link            | [Models🤗](https://huggingface.co/StarJiaxing/R1-Omni-0.5B)  [Code💻](https://github.com/HumanMLLM/R1-Omni) |
| Reward Function | Rule-based Rewards (Accuracy, Format)                        |
| Core Insights   | Apply the r1 training paradigm to the field of audio-video emotion recognition. |

**EchoInk-R1: Exploring Audio-Visual Reasoning in Multimodal LLMs via Reinforcement Learning**

| Paper           | [EchoInk-R1: Exploring Audio-Visual Reasoning in Multimodal LLMs via Reinforcement Learning](https://arxiv.org/abs/2505.04623) |
| :-------------- | :----------------------------------------------------------- |
| Link            | [Datasets🤗](https://huggingface.co/datasets/harryhsing/AVQA-R1-6K)   [Models🤗](https://huggingface.co/harryhsing/EchoInk-R1-7B)   [Code💻](https://github.com/HarryHsing/EchoInk) |
| Reward Function | Answer Accuracy, Format Consistency                          |
| Core Insights   | The authors introduce EchoInk-R1, a reinforcement learning framework designed to enhance audio-visual reasoning in multimodal large language models (MLLMs). By leveraging GRPO and task-specific rewards, EchoInk-R1 achieves significant performance gains with minimal training iterations. Notably, the model demonstrates "aha moments," self-corrective reasoning behaviors that arise when revisiting initial assumptions under ambiguity, showcasing cross-modal reflective reasoning capabilities. |

</details>

# Benchmarks 📈

| Benchmark                                                    | Date | Org             | Modality | Applications         |
| ------------------------------------------------------------ | ---- | --------------- | -------- | -------------------- |
| [MME-CoT](https://arxiv.org/abs/2502.00698)                  | 2.02 | Tencent         | T&I      | Multi-Step Reasoning |
| [ZeroBench](https://arxiv.org/abs/2502.09696)                | 2.13 | Cambridge       | T&I      | Multi-Step Reasoning |
| [MDK12-Bench](https://github.com/Sun-Haoyuan23/Awesome-RL-based-Reasoning-MLLMs) | 4.08 | Shanghai AI Lab | T&I      | Multi-Step Reasoning |
| [VCR-Bench](https://arxiv.org/abs/2504.07956)                | 4.10 | USTC            | T&V      | Multi-Step Reasoning |
| [GeoSense](https://arxiv.org/abs/2504.12597)                 | 4.17 | Alibaba         | T&I      | Geometry Reasoning   |
| [Video-MMLU](https://arxiv.org/abs/2504.14693)               | 4.20 | ZJU             | T&V      | Multi-Step Reasoning |
| [VisuLogic](https://arxiv.org/abs/2504.15279)                | 4.21 | Shanghai AI Lab | T&I      | Multi-Step Reasoning |
| [GDI-Bench](https://www.arxiv.org/abs/2505.00063)            | 4.30 | Shanghai AI Lab | T&I      | Multi-Step Reasoning |

**MM-IQ: Benchmarking Human-Like Abstraction and Reasoning in Multimodal Models**

| Paper         | [MM-IQ: Benchmarking Human-Like Abstraction and Reasoning in Multimodal Models](https://arxiv.org/abs/2502.00698) |
| :------------ | :----------------------------------------------------------- |
| Link          | [Project🎯](https://acechq.github.io/MMIQ-benchmark/)  [Datasets🤗](https://acechq.github.io/MMIQ-benchmark/)   [Code💻](https://acechq.github.io/MMIQ-benchmark/) |
| Core Insights | MM-IQ is a multimodal benchmark comprising 2,710 visual reasoning problems across 8 distinct paradigms (e.g., logical operations, 2D/3D geometry, spatial relationships). It evaluates MLLMs' abstraction and reasoning capabilities, focusing on high-level cognitive abilities without linguistic or domain-specific biases. |

<details><summary>Show More...</summary>

**ZeroBench: An Impossible Visual Benchmark for Contemporary Large Multimodal Models**

| Paper         | [ZeroBench: An Impossible Visual Benchmark for Contemporary Large Multimodal Models](https://arxiv.org/abs/2502.09696) |
| :------------ | :----------------------------------------------------------- |
| Link          | [Project🎯](https://zerobench.github.io/)  [Datasets🤗](https://huggingface.co/datasets/jonathan-roberts1/zerobench)   [Code💻](https://github.com/jonathan-roberts1/zerobench/) |
| Core Insights | ZeroBench is a lightweight visual reasoning benchmark consisting of 100 hand-crafted questions and 334 subquestions, designed to evaluate the complex visual reasoning capabilities of Large Multimodal Models (LMMs). The benchmark focuses on multi-step reasoning requiring precise, exact answers that cannot be easily guessed, covering both natural and synthetic images. All evaluated models scored 0.0% on the main questions, highlighting significant challenges in visual interpretation. |

**MDK12-Bench: A Multi-Discipline Benchmark for Evaluating Reasoning in Multimodal Large Language Models**

| Paper         | [MDK12-Bench: A Multi-Discipline Benchmark for Evaluating Reasoning in Multimodal Large Language Models](https://arxiv.org/abs/2504.05782) |
| :------------ | :----------------------------------------------------------- |
| Link          | [Code💻](https://github.com/LanceZPF/MDK12)                   |
| Core Insights | The benchmark is multimodal, encompassing text and images, and includes diverse question types like multiple-choice, fill-in-the-blank, true/false, and open-ended queries. It evaluates MLLMs on reasoning capabilities across six K-12 academic disciplines (mathematics, physics, chemistry, biology, geography, and information science), assessing knowledge coverage, difficulty handling, and robustness to dynamic perturbations via a novel bootstrapping framework. |

**VCR-Bench: A Comprehensive Evaluation Framework for Video Chain-of-Thought Reasoning**

| Paper         | [VCR-Bench: A Comprehensive Evaluation Framework for Video Chain-of-Thought Reasoning](https://arxiv.org/abs/2504.07956) |
| :------------ | :----------------------------------------------------------- |
| Link          | [Project🎯](https://vlm-reasoning.github.io/VCR-Bench/)  [Datasets🤗](https://huggingface.co/datasets/VLM-Reasoning/VCR-Bench)   [Code💻](https://github.com/zhishuifeiqian/VCR-Bench) |
| Core Insights | VCR-Bench is a video modality benchmark designed to evaluate the Chain-of-Thought (CoT) reasoning capabilities of Large Vision-Language Models (LVLMs). It includes 859 videos and 1,034 question-answer pairs across seven task dimensions. The benchmark assesses both perception and logical reasoning steps through metrics like Recall, Precision, and F1 score, aiming to expose performance bottlenecks in temporal-spatial information processing and reasoning. |

**GeoSense: Evaluating Identification and Application of Geometric Principles in Multimodal Reasoning**

| Paper         | [GeoSense: Evaluating Identification and Application of Geometric Principles in Multimodal Reasoning](https://arxiv.org/abs/2504.12597) |
| :------------ | :----------------------------------------------------------- |
| Link          | -                                                            |
| Core Insights | GeoSense is a bilingual (English and Chinese) benchmark focusing on geometry problem-solving (GPS). It evaluates multimodal large language models (MLLMs) on their ability to identify and apply geometric principles within visual diagrams. The benchmark introduces two key metrics: Geometric Principles Identification (GPI) and Geometric Principles Application (GPA), assessing both the recognition of necessary geometric concepts and their correct usage in complex visual contexts. |

**Video-MMLU: A Massive Multi-Discipline Lecture Understanding Benchmark**

| Paper         | [Video-MMLU: A Massive Multi-Discipline Lecture Understanding Benchmark](https://arxiv.org/abs/2504.14693) |
| :------------ | :----------------------------------------------------------- |
| Link          | [Project🎯](https://enxinsong.com/Video-MMLU-web/)  [Datasets🤗](https://huggingface.co/datasets/Enxin/Video-MMLU)   [Code💻](https://github.com/Espere-1119-Song/Video-MMLU) |
| Core Insights | Video modality with multi-discipline lecture content; designed for detailed captioning and reasoning QA tasks to evaluate MLLMs on visual perception, comprehension, and reasoning across mathematics, physics, and chemistry. |

**VisuLogic: A Benchmark for Evaluating Visual Reasoning in Multi-modal Large Language Models**

| Paper         | [VisuLogic: A Benchmark for Evaluating Visual Reasoning in Multi-modal Large Language Models](https://arxiv.org/abs/2504.15279) |
| :------------ | :----------------------------------------------------------- |
| Link          | [Project🎯](https://visulogic-benchmark.github.io/VisuLogic/)  [Datasets🤗](https://huggingface.co/datasets/VisuLogic/VisuLogic)   [Code💻](https://github.com/VisuLogic-Benchmark) |
| Core Insights | VisuLogic is a multimodal benchmark comprising 1,000 vision-centric reasoning tasks across six categories (e.g., quantitative shifts, spatial relations). It evaluates MLLMs' genuine visual reasoning capabilities without relying on text-based shortcuts, focusing on robust and in-depth visual inference skills. |

**GDI-Bench: A Benchmark for General Document Intelligence with Vision and Reasoning Decoupling**

| Paper         | [GDI-Bench: A Benchmark for General Document Intelligence with Vision and Reasoning Decoupling](https://www.arxiv.org/abs/2505.00063) |
| :------------ | :----------------------------------------------------------- |
| Link          | -                                                            |
| Core Insights | Multimodal (vision and text), designed tasks across visual complexity and reasoning difficulty to evaluate MLLMs' document understanding, OCR, and reasoning capabilities. |

</details>

# Contribution and Acknowledgment ❤️

This is an active repository, and your contributions are always welcome! If you have any questions, please feel free to contact [ghzhou@stu.ecnu.edu.cn](ghzhou@stu.ecnu.edu.cn) or [panjiaqiu@stu.ecnu.edu.cn](panjiaqiu@stu.ecnu.edu.cn).

I sincerely thank all community members who have provided valuable supplementary support.

# Citation 📄

If you find this repository useful for your research and applications, please star us ⭐ and consider citing:

```text
@article{zhou2025reinforced,
  title={Reinforced MLLM: A Survey on RL-Based Reasoning in Multimodal Large Language Models},
  author={Zhou, Guanghao and Qiu, Panjia and Chen, Cen and Wang, Jie and Yang, Zheming and Xu, Jian and Qiu, Minghui},
  journal={arXiv preprint arXiv:2504.21277},
  year={2025}
}
```