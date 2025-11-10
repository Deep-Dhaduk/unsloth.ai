# 🚀 Unsloth.ai Training Techniques - Modern AI Assignment

> **A comprehensive collection of 5 Jupyter notebooks demonstrating advanced language model training methodologies using Unsloth.ai framework**

---

## � What's Inside This Repository?

This repository provides **5 production-ready notebooks**, each focusing on a distinct training paradigm for large language models. Every notebook includes comprehensive documentation, working code examples, and detailed explanations suitable for both learning and video demonstrations.

---

## 🗂️ Complete Notebook Collection

| # | Notebook File | Training Method | Key Focus |
|---|--------------|-----------------|-----------|
| 1 | `colab1_full_finetuning_smollm2.ipynb` | Full Parameter Finetuning | Train all 135M parameters |
| 2 | `colab2_lora_finetuning_smollm2.ipynb` | LoRA Adapters | Parameter-efficient training |
| 3 | `colab3_dpo_reinforcement_learning.ipynb` | Direct Preference Optimization | Alignment from human feedback |
| 4 | `colab4_grpo_reasoning_model.ipynb` | Group Relative Policy Optimization | Self-improving reasoning |
| 5 | `colab5_continued_pretraining.ipynb` | Continued Pretraining | Domain knowledge adaptation |

---

## 📖 Detailed Notebook Descriptions

### 🔹 Notebook 1: Full Parameter Finetuning
**File:** `colab1_full_finetuning_smollm2.ipynb`

**Training Approach:**
- Updates every single parameter in the model (all 135 million parameters)
- Trains SmolLM2-135M-Instruct from HuggingFace
- Uses Alpaca cleaned dataset for instruction-following tasks
- Implements gradient checkpointing for memory optimization

**Core Concepts Covered:**
- Full finetuning methodology vs parameter-efficient methods
- Gradient checkpointing techniques for large model training
- Chat template formatting for conversational AI
- Memory management strategies for training

**Training Data:** `yahma/alpaca-cleaned` - instruction-response pairs for teaching models to follow instructions

---

### 🔹 Notebook 2: LoRA Finetuning
**File:** `colab2_lora_finetuning_smollm2.ipynb`

**Training Approach:**
- Trains only adapter layers (~1-2% of total parameters)
- Same base model (SmolLM2-135M-Instruct) as Notebook 1
- Same dataset for direct performance comparison
- Dramatically reduces memory requirements

**Core Concepts Covered:**
- Low-Rank Adaptation (LoRA) theory and implementation
- LoRA rank (r) parameter selection and alpha configuration
- Target module selection strategy
- Adapter merging and deployment

**Training Data:** `yahma/alpaca-cleaned` - identical to Notebook 1 for comparative analysis

**Critical Configuration:** Setting `r=16` enables LoRA mode (not `r=0` which would be full finetuning)

---

### 🔹 Notebook 3: DPO Reinforcement Learning
**File:** `colab3_dpo_reinforcement_learning.ipynb`

**Training Approach:**
- Implements Direct Preference Optimization algorithm
- Learns from preference pairs (chosen vs rejected responses)
- Aligns model behavior with human preferences
- Eliminates need for separate reward model

**Core Concepts Covered:**
- DPO beta hyperparameter tuning
- Preference dataset formatting requirements
- Alignment training methodology
- Human feedback integration

**Training Data:** `Intel/orca_dpo_pairs` - contains triplets of (prompt, chosen_response, rejected_response)

**Required Data Structure:**
```json
{
  "prompt": "User question or instruction",
  "chosen": "High-quality preferred response",
  "rejected": "Lower-quality rejected response"
}
```

---

### 🔹 Notebook 4: GRPO Reasoning Model
**File:** `colab4_grpo_reasoning_model.ipynb`

**Training Approach:**
- Group Relative Policy Optimization for reasoning tasks
- Self-improvement through multiple solution attempts
- Automatic evaluation using reward functions
- Builds reasoning capabilities similar to OpenAI's o1 model

**Core Concepts Covered:**
- Multiple candidate solution generation
- Reward function design and implementation
- Self-supervised learning paradigm
- Advanced reasoning skill development

**Training Data:** Mathematically generated problems for reasoning tasks

**Training Pipeline:**
1. Model generates multiple solution attempts
2. Reward function evaluates each solution
3. Model learns from successful solutions
4. Reasoning capability improves iteratively

---

### 🔹 Notebook 5: Continued Pretraining
**File:** `colab5_continued_pretraining.ipynb`

**Training Approach:**
- Extends pretrained model to new knowledge domains
- Trains on raw, unformatted domain-specific text
- Expands model's vocabulary and domain understanding
- Adapts existing knowledge to specialized fields

**Core Concepts Covered:**
- Continued pretraining vs initial pretraining
- Domain adaptation strategies
- Raw text processing and training
- Knowledge expansion without catastrophic forgetting

**Training Data:** `code_search_net` - Python source code for programming domain adaptation

**Primary Applications:**
- Medical and legal domain specialization
- Programming language learning
- Multilingual capability expansion
- Incorporating recent knowledge and events

---

## ⚡ Getting Started

### 🎯 Two Paths to Run These Notebooks

#### **Path 1: Google Colab** *(Recommended - Zero Setup)*

Google Colab is the fastest way to get started with no installation required.

**Requirements:**
- Google account with Colab access
- Free T4 GPU tier is sufficient for all notebooks
- No local installation necessary

**Steps to Execute:**
1. Navigate to [Google Colab](https://colab.research.google.com/)
2. Upload any of the 5 `.ipynb` notebook files
3. Configure GPU access: `Runtime` → `Change runtime type` → Select `GPU`
4. Execute cells sequentially from top to bottom
5. Each notebook is completely self-contained

**Training Duration on Colab T4 GPU:**
- Full Finetuning (Notebook 1): Approximately 10-15 minutes
- LoRA Finetuning (Notebook 2): Approximately 5-10 minutes
- DPO Training (Notebook 3): Approximately 10-15 minutes
- GRPO Training (Notebook 4): Approximately 15-20 minutes
- Continued Pretraining (Notebook 5): Approximately 15-20 minutes

---

#### **Path 2: Local Environment** *(Advanced Users)*

For users with local GPU infrastructure or those wanting offline access.

**System Requirements:**
- Python 3.8 or higher installed
- pip package manager
- NVIDIA GPU with CUDA support (recommended)

**Quick Setup Commands:**

```bash
# Windows PowerShell users:
.\setup_venv.ps1

# Linux / macOS / Git Bash users:
./setup_venv.sh
```

**Manual Setup Process:**

**Step 1 - Create Virtual Environment:**
```bash
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# On Linux/macOS:
source venv/bin/activate
```

**Step 2 - Launch Notebook Interface:**
```bash
# Option A: Start Jupyter Notebook server
jupyter notebook

# Option B: Open in VS Code with Jupyter extension
code .
```

**Step 3 - Select Python Kernel:**
- **Jupyter Interface:** Navigate to `Kernel` → `Change kernel` → Select `venv`
- **VS Code:** Click kernel selector → `Python Environments` → Select `venv`

**Performance Expectations:**
- **With NVIDIA GPU:** Performance comparable to Google Colab
- **CPU Only:** Training will be 5-10x slower (not recommended for production use)

📘 **For comprehensive setup instructions, refer to:** [VENV_SETUP.md](VENV_SETUP.md)

---

## 📊 Notebook Comparison Matrix

| Training Method | Trainable Parameters | Memory Footprint | Training Speed | Optimal Use Cases |
|----------------|---------------------|------------------|----------------|-------------------|
| **Notebook 1** - Full Finetuning | ALL 135M (100%) | High | Slower | Mission-critical applications |
| **Notebook 2** - LoRA | ~1-2M (1-2%) | Low | Faster | General-purpose tasks |
| **Notebook 3** - DPO | ~1-2M (LoRA-based) | Medium | Medium | Model alignment & safety |
| **Notebook 4** - GRPO | ~1-2M (LoRA-based) | Medium | Medium | Complex reasoning tasks |
| **Notebook 5** - CPT | ~1-2M (LoRA-based) | Medium | Medium | Domain specialization |

---

## 🎓 Recommended Learning Sequence

### Optimal Study Path for Maximum Understanding:

**📍 Stage 1:** Begin with **Notebook 2 (LoRA Finetuning)**
- Why: Shortest training time, easiest to understand
- Learn: Parameter-efficient training fundamentals

**📍 Stage 2:** Progress to **Notebook 1 (Full Finetuning)**
- Why: Direct comparison with LoRA approach
- Learn: Trade-offs between efficiency and performance

**📍 Stage 3:** Advance to **Notebook 5 (Continued Pretraining)**
- Why: Understand domain adaptation techniques
- Learn: How to teach models new knowledge domains

**📍 Stage 4:** Explore **Notebook 3 (DPO)**
- Why: Grasp model alignment concepts
- Learn: Human preference learning mechanisms

**📍 Stage 5:** Master **Notebook 4 (GRPO)**
- Why: Most advanced reasoning techniques
- Learn: Self-improving AI systems

---

## 🧠 Fundamental Concepts

### Training Methodology Hierarchy:
```
Pretraining → Continued Pretraining → Finetuning → Alignment
     ↓                 ↓                   ↓            ↓
Learn Language    Acquire Domain      Format Tasks   User Preferences
  Structure         Knowledge          & Skills         & Safety
```

### Parameter Efficiency Comparison:
```
Full Finetuning:  Updates 100% of parameters (135M for SmolLM2)
LoRA Adaptation:  Updates 1-2% of parameters (~1-2M adapter weights)
```

### Dataset Format Requirements:

**Supervised Fine-Tuning (SFT):**
```
Structure: Prompt + Response
Example: {"instruction": "...", "response": "..."}
```

**Direct Preference Optimization (DPO):**
```
Structure: Prompt + Chosen + Rejected
Example: {"prompt": "...", "chosen": "...", "rejected": "..."}
```

**Group Relative Policy Optimization (GRPO):**
```
Structure: Prompt only
Process: Model generates multiple responses, learns from rewards
```

**Continued Pretraining (CPT):**
```
Structure: Raw unformatted text
Example: Plain text documents without instruction formatting
```

---

## 🎬 Video Documentation Guidelines

### Complete Video Structure (Per Notebook)

#### **Segment 1: Introduction & Context** *(2-3 minutes)*
- Identify the specific training technique being demonstrated
- Explain the importance and real-world applications
- Preview the demonstration objectives

#### **Segment 2: Code Implementation Walkthrough** *(10-15 minutes)*
- Execute dependency installation commands
- Load and initialize the model and dataset
- Explain critical hyperparameters and their purposes
- Configure training settings and arguments
- Run the complete training loop
- Display and interpret output results

#### **Segment 3: Technical Deep Dive** *(5-10 minutes)*

**For Notebook 1 (Full Finetuning):**
- Explain full parameter training methodology
- Discuss when full finetuning is necessary vs overkill

**For Notebook 2 (LoRA):**
- Describe Low-Rank Adaptation mechanism
- Demonstrate memory savings and efficiency gains

**For Notebook 3 (DPO):**
- Explain Direct Preference Optimization algorithm
- Show how human feedback aligns model behavior

**For Notebook 4 (GRPO):**
- Describe Group Relative Policy Optimization
- Explain reasoning model training dynamics

**For Notebook 5 (CPT):**
- Explain continued pretraining methodology
- Demonstrate domain knowledge acquisition

#### **Segment 4: Data Format Demonstration** *(3-5 minutes)*
- Display concrete dataset examples
- Clarify format specifications and requirements
- Walk through preprocessing and transformation steps

#### **Segment 5: Results Analysis & Inference** *(5-10 minutes)*
- Present training metrics and loss curves
- Execute inference with diverse examples
- Compare model performance before and after training
- Discuss observable improvements and limitations

#### **Segment 6: Best Practices & Recommendations** *(2-3 minutes)*
- Share hyperparameter tuning strategies
- Highlight common pitfalls and mistakes
- Provide actionable recommendations

**Expected Total Duration:** 25-40 minutes per notebook

---

## ✅ Submission Verification Checklist

### Required Deliverables (5 Complete Sets)

**🔷 Notebook 1 Deliverables:**
- [ ] Full finetuning notebook file
- [ ] Comprehensive video demonstration

**🔷 Notebook 2 Deliverables:**
- [ ] LoRA finetuning notebook file
- [ ] Comprehensive video demonstration

**🔷 Notebook 3 Deliverables:**
- [ ] DPO alignment notebook file
- [ ] Comprehensive video demonstration

**🔷 Notebook 4 Deliverables:**
- [ ] GRPO reasoning notebook file
- [ ] Comprehensive video demonstration

**🔷 Notebook 5 Deliverables:**
- [ ] Continued pretraining notebook file
- [ ] Comprehensive video demonstration

### Quality Assurance Criteria (For Each Notebook):
- [ ] Successfully executes from start to finish
- [ ] Zero cell execution errors
- [ ] All training results properly displayed
- [ ] Video includes complete code walkthrough
- [ ] Explanations are clear and technically accurate

---

## 🔧 Common Issues & Solutions

### Problem 1: Out of Memory (OOM) Errors

**Symptoms:** Training crashes with CUDA out of memory error

**Solutions:**
```python
# Solution A: Reduce batch size
per_device_train_batch_size = 1  # Lower from default 2

# Solution B: Reduce sequence length
max_seq_length = 1024  # Lower from default 2048

# Solution C: Enable gradient checkpointing
use_gradient_checkpointing = "unsloth"
```

---

### Problem 2: Excessively Slow Training

**Symptoms:** Training takes much longer than expected

**Solutions:**
```python
# Solution A: Enable sequence packing
packing = True

# Solution B: Reduce training steps
max_steps = 30  # Lower from default 60

# Solution C: Use smaller dataset subset
dataset = dataset.select(range(500))  # Train on first 500 samples only
```

---

### Problem 3: Model Not Learning Effectively

**Symptoms:** Loss plateaus or doesn't decrease

**Solutions:**
```python
# Solution A: Increase learning rate (for LoRA/CPT)
learning_rate = 5e-4  # Increase from default

# Solution B: Increase LoRA rank for more capacity
r = 32  # Increase from default 16

# Solution C: Extend training duration
max_steps = 200  # Increase from default
```

---

### Problem 4: CUDA Memory Management Issues

**Symptoms:** Inconsistent memory errors or fragmentation

**Solutions:**
```python
# Solution A: Clear GPU cache before training
import torch
torch.cuda.empty_cache()

# Solution B: Adjust batch accumulation strategy
per_device_train_batch_size = 1
gradient_accumulation_steps = 8
```

---

## 💡 Success Strategies

### Strategy 1: Progressive Complexity Approach
**Start Small, Scale Gradually:**
- Begin with compact models: SmolLM2-135M or Llama-3.2-1B
- Use limited datasets: Start with 1000 samples
- Run short training cycles: 60 steps initially
- Incrementally increase complexity after validation

### Strategy 2: Comprehensive Monitoring
**Track Everything:**
- Continuously monitor loss curve trajectories
- Watch GPU memory utilization in real-time
- Measure and record training duration for each run
- Implement checkpoint saving at regular intervals

### Strategy 3: Rigorous Testing Protocol
**Validate Thoroughly:**
- Conduct before-and-after performance comparisons
- Test with diverse input scenarios
- Use varied prompt formulations
- Evaluate edge cases and boundary conditions

### Strategy 4: Detailed Documentation
**Record Your Journey:**
- Capture screenshots of key results
- Save all training outputs and logs
- Document metrics systematically
- Record observations and insights

### Strategy 5: Deep Parameter Understanding
**Learn, Don't Copy:**
- Avoid blindly copying configuration values
- Experiment with parameter variations
- Understand the impact of each hyperparameter
- Iteratively tune to find optimal settings for your use case

---

## 🌟 Advanced Exploration Opportunities

### Extension 1: Multi-Stage Training Pipeline
**Combine multiple techniques sequentially:**
```
Stage 1: CPT (Domain Knowledge)
   ↓
Stage 2: SFT (Task Formatting)
   ↓
Stage 3: DPO (Preference Alignment)
```

### Extension 2: Model Deployment with Ollama
**Export models for production use:**
```python
# Quantize and export model
model.save_pretrained_gguf(
    "model_name",
    tokenizer,
    quantization_method="q4_k_m"
)
```

### Extension 3: Sequential Training Stages
**Implement comprehensive training workflow:**
```python
# Stage 1: Acquire domain knowledge through continued pretraining
# Stage 2: Learn task-specific formatting via instruction finetuning
# Stage 3: Align outputs with user preferences using DPO
```

### Extension 4: Custom Dataset Development
**Create specialized training data:**
- Curate domain-specific text corpora
- Gather and annotate preference pairs
- Design custom reward functions for GRPO
- Build specialized models for niche applications

### Extension 5: Larger Model Experimentation
**Scale to more powerful architectures:**
- Llama-3.1-8B (8 billion parameters)
- Phi-3.5-mini (optimized efficiency)
- Mistral-7B (strong general performance)
- ⚠️ Note: Requires significantly more GPU memory (16GB+ VRAM)

---

## 📚 Essential Resources & References

### Official Unsloth Documentation
- **[Complete Documentation](https://docs.unsloth.ai/)** - Comprehensive framework guide
- **[GitHub Repository](https://github.com/unslothai/unsloth)** - Source code and issues
- **[Example Notebooks](https://github.com/unslothai/notebooks)** - Additional tutorials

### Training Methodology Guides
- **[LLM Fine-tuning Guide](https://docs.unsloth.ai/get-started/fine-tuning-llms-guide)** - Comprehensive finetuning tutorial
- **[Reinforcement Learning Guide](https://docs.unsloth.ai/get-started/reinforcement-learning-rl-guide)** - RL training methods
- **[Continued Pretraining Guide](https://docs.unsloth.ai/basics/continued-pretraining)** - Domain adaptation techniques

### Dataset Repositories
- **[Hugging Face Datasets Hub](https://huggingface.co/datasets)** - Central dataset repository
- **[Alpaca Cleaned Dataset](https://huggingface.co/datasets/yahma/alpaca-cleaned)** - Instruction-following data
- **[Orca DPO Pairs](https://huggingface.co/datasets/Intel/orca_dpo_pairs)** - Preference pair examples
- **[Code Search Net](https://huggingface.co/datasets/code_search_net)** - Programming language datasets

---

## 🎯 Assignment Evaluation Framework

### Assessment Criteria Breakdown

**Component 1: Code Execution Quality (40% Weight)**
- All 5 notebooks execute successfully from start to finish
- Zero runtime errors or exceptions
- All cells produce expected outputs
- Training completes and results are displayed

**Component 2: Video Demonstration Quality (30% Weight)**
- Clear and articulate explanations throughout
- Comprehensive code walkthrough with context
- Demonstrates inputs, processing, and outputs
- Explains underlying concepts and methodology

**Component 3: Technical Understanding (20% Weight)**
- Articulates how each training method functions
- Demonstrates understanding of dataset format requirements
- Shows knowledge of hyperparameter impacts
- Identifies appropriate use cases for each technique

**Component 4: Submission Completeness (10% Weight)**
- All 5 notebooks submitted
- All 5 video demonstrations included
- Proper documentation and comments
- Timely submission within deadline

---

## 🤝 Support & Assistance

### Self-Service Resources (Recommended First Steps)

1. **Notebook Inline Comments**
   - Each cell contains detailed explanatory comments
   - Technical rationale provided for all major decisions

2. **Summary Sections**
   - Every notebook includes comprehensive summary sections
   - Key concepts and takeaways clearly documented

3. **Official Documentation**
   - Links to authoritative Unsloth resources provided
   - Follow references for deeper technical details

4. **Hands-On Experimentation**
   - Modify parameters and observe outcomes
   - Learning through experimentation is most effective

---

## 📄 Repository Summary

### What You Have Access To:

This repository provides **5 production-ready training notebooks**, each demonstrating a distinct approach:

✅ **Full Parameter Finetuning** - Complete model parameter training  
✅ **LoRA Adapter Training** - Parameter-efficient adaptation  
✅ **DPO Preference Alignment** - Human feedback integration  
✅ **GRPO Reasoning Enhancement** - Self-improving reasoning systems  
✅ **Continued Pretraining** - Domain knowledge acquisition  

### Quality Assurance:

Each notebook guarantees:
- ✅ Complete functionality with zero configuration required
- ✅ Extensive inline documentation and explanations
- ✅ Working code examples that execute successfully
- ✅ Best practices and optimization strategies
- ✅ Structure optimized for video demonstration

**Best of luck with your assignment! 🚀**

---

## 📞 Getting Help

### When You Need Assistance:

**Step 1:** Examine the detailed comments within each notebook cell  
**Step 2:** Review the comprehensive summary sections in each notebook  
**Step 3:** Consult the official Unsloth documentation linked throughout  

### Learning Philosophy:

**💡 Master through practice:** The most effective learning comes from executing code, experimenting with hyperparameters, and understanding the mechanics of each training technique!
