# Modern AI with Unsloth.ai - Complete Assignment

## 📋 Assignment Overview

This repository contains 5 comprehensive Jupyter notebooks covering modern AI training techniques using Unsloth.ai. Each notebook demonstrates a different training approach with detailed explanations, code, and examples.

## 🎯 Notebooks Created

### 1. Colab 1: Full Finetuning with SmolLM2 (`colab1_full_finetuning_smollm2.ipynb`)
**What it does:**
- Full parameter finetuning (all 135M parameters updated)
- Uses SmolLM2-135M-Instruct model
- Trains on Alpaca instruction dataset
- Demonstrates complete model training

**Key Concepts:**
- Full finetuning vs LoRA
- Gradient checkpointing
- Chat template formatting
- Memory-intensive training

**Dataset:** `yahma/alpaca-cleaned` (instruction-following)

---

### 2. Colab 2: LoRA Finetuning with SmolLM2 (`colab2_lora_finetuning_smollm2.ipynb`)
**What it does:**
- Parameter-efficient finetuning with LoRA
- Only updates ~1-2% of parameters
- Uses same model and dataset as Colab 1
- Demonstrates LoRA adapters

**Key Concepts:**
- LoRA rank (r) and alpha parameters
- Target modules selection
- Adapter-based training
- Memory efficiency

**Dataset:** Same as Colab 1 for direct comparison

**Key Difference:** `r=16` (not 0) enables LoRA mode

---

### 3. Colab 3: DPO Reinforcement Learning (`colab3_dpo_reinforcement_learning.ipynb`)
**What it does:**
- Direct Preference Optimization (DPO)
- Learns from human preferences
- Aligns model with chosen vs rejected responses
- No reward model needed

**Key Concepts:**
- DPO beta parameter
- Preference pair formatting
- Alignment training
- Human feedback learning

**Dataset:** `Intel/orca_dpo_pairs` (prompt + chosen + rejected)

**Format Required:**
```json
{
  "prompt": "Question",
  "chosen": "Good response",
  "rejected": "Bad response"
}
```

---

### 4. Colab 4: GRPO Reasoning Model (`colab4_grpo_reasoning_model.ipynb`)
**What it does:**
- Group Relative Policy Optimization (GRPO)
- Trains reasoning models (like o1)
- Self-improvement through trial and error
- Automatic reward evaluation

**Key Concepts:**
- Multiple solution generation
- Reward function design
- Self-supervised learning
- Reasoning capability

**Dataset:** Math problems (generated)

**How it works:**
1. Generate multiple solutions
2. Evaluate with reward function
3. Learn from correct solutions
4. Improve reasoning ability

---

### 5. Colab 5: Continued Pretraining (`colab5_continued_pretraining.ipynb`)
**What it does:**
- Teaches model new domain knowledge
- Pretraining on raw domain text
- Expands vocabulary and knowledge
- Domain adaptation

**Key Concepts:**
- Continued vs initial pretraining
- Raw text training
- Domain expertise
- Knowledge expansion

**Dataset:** `code_search_net` (Python code)

**Use Cases:**
- Medical/Legal domain
- New programming languages
- Multilingual capabilities
- Recent events/knowledge

---

## 🚀 Quick Start Guide

### Prerequisites

**Option 1: Google Colab (Recommended for Beginners)**
```bash
# You'll need a Google Colab account with GPU access
# T4 GPU (free tier) is sufficient for all notebooks
# No local setup required!
```

**Option 2: Local Setup with Virtual Environment**
```bash
# Python 3.8+ with pip installed
# See VENV_SETUP.md for detailed instructions

# Quick setup:
.\setup_venv.ps1  # Windows
./setup_venv.sh   # Linux/Mac
```

### Running the Notebooks

**Option A: Google Colab (Easiest)**

1. **Upload to Google Colab:**
   - Go to [Google Colab](https://colab.research.google.com/)
   - Upload each `.ipynb` file
   - Enable GPU: Runtime → Change runtime type → GPU

2. **Run Cells Sequentially:**
   - Each notebook is self-contained
   - Run cells from top to bottom
   - Follow the explanations

**Option B: Local Environment (Advanced)**

1. **Setup Virtual Environment:**
   ```bash
   # Windows PowerShell
   .\setup_venv.ps1
   
   # Linux/Mac/Git Bash
   ./setup_venv.sh
   ```

2. **Activate Environment:**
   ```bash
   # Windows
   .\venv\Scripts\Activate.ps1
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Open Jupyter or VS Code:**
   ```bash
   # Start Jupyter Notebook
   jupyter notebook
   
   # Or use VS Code with Jupyter extension
   code .
   ```

4. **Select Kernel:**
   - In Jupyter: Kernel → Change kernel → venv
   - In VS Code: Select kernel → Python Environments → venv

📖 **Detailed setup instructions:** See [VENV_SETUP.md](VENV_SETUP.md)

### Expected Runtime

**Google Colab (T4 GPU):**
   - Colab 1 (Full Finetuning): ~10-15 minutes
   - Colab 2 (LoRA): ~5-10 minutes
   - Colab 3 (DPO): ~10-15 minutes
   - Colab 4 (GRPO): ~15-20 minutes
   - Colab 5 (CPT): ~15-20 minutes

**Local (varies by GPU):**
   - With NVIDIA GPU: Similar to Colab
   - CPU only: 5-10x slower (not recommended)

---

## 📊 Comparison Table

| Notebook | Method | Parameters Updated | Memory | Speed | Use Case |
|----------|--------|-------------------|--------|-------|----------|
| Colab 1 | Full Finetuning | ALL (135M) | High | Slower | Critical tasks |
| Colab 2 | LoRA | ~1-2M (1-2%) | Low | Faster | Most tasks |
| Colab 3 | DPO | ~1-2M (LoRA) | Medium | Medium | Alignment |
| Colab 4 | GRPO | ~1-2M (LoRA) | Medium | Medium | Reasoning |
| Colab 5 | CPT | ~1-2M (LoRA) | Medium | Medium | New domains |

---

## 🎓 Learning Path

### Recommended Order:
1. **Start with Colab 2** (LoRA) - Easiest and fastest
2. **Then Colab 1** (Full Finetuning) - Compare differences
3. **Move to Colab 5** (CPT) - Learn domain adaptation
4. **Try Colab 3** (DPO) - Understand alignment
5. **Finally Colab 4** (GRPO) - Advanced reasoning

### Key Concepts to Understand:

**Training Types:**
```
Pretraining → Continued Pretraining → Finetuning → Alignment
    ↓                ↓                    ↓            ↓
Learn language    New domain         Format      Preferences
```

**Parameter Efficiency:**
```
Full Finetuning: 100% parameters
LoRA: 1-2% parameters (adapters)
```

**Dataset Formats:**
```
SFT: Prompt + Response
DPO: Prompt + Chosen + Rejected
GRPO: Prompt only (generates responses)
CPT: Raw text (no structure)
```

---

## 🎬 Video Recording Guide

For each notebook, record a video covering:

### 1. Introduction (2-3 minutes)
- What technique you're demonstrating
- Why it's important
- What you'll show

### 2. Code Walkthrough (10-15 minutes)
- Install dependencies
- Load model and data
- Explain key parameters
- Show training configuration
- Run training
- Display results

### 3. Key Explanations (5-10 minutes)
- **Colab 1:** What is full finetuning? When to use it?
- **Colab 2:** What is LoRA? How does it save memory?
- **Colab 3:** What is DPO? How does it align models?
- **Colab 4:** What is GRPO? How do reasoning models work?
- **Colab 5:** What is CPT? How to teach new knowledge?

### 4. Dataset Format (3-5 minutes)
- Show example data
- Explain format requirements
- Demonstrate preprocessing

### 5. Results & Inference (5-10 minutes)
- Show training metrics
- Run inference examples
- Compare before/after
- Discuss improvements

### 6. Tips & Best Practices (2-3 minutes)
- Parameter tuning advice
- Common mistakes
- Recommendations

**Total per video:** 25-40 minutes

---

## 📝 Assignment Submission Checklist

- [ ] **Colab 1:** Full finetuning notebook + video
- [ ] **Colab 2:** LoRA finetuning notebook + video  
- [ ] **Colab 3:** DPO alignment notebook + video
- [ ] **Colab 4:** GRPO reasoning notebook + video
- [ ] **Colab 5:** Continued pretraining notebook + video

For each:
- [ ] Notebook runs successfully
- [ ] All cells execute without errors
- [ ] Results are shown
- [ ] Video shows full walkthrough
- [ ] Explanations are clear

---

## 🔧 Troubleshooting

### Common Issues:

**1. Out of Memory Error:**
```python
# Reduce batch size
per_device_train_batch_size=1  # instead of 2

# Reduce sequence length
max_seq_length=1024  # instead of 2048

# Use gradient checkpointing
use_gradient_checkpointing="unsloth"
```

**2. Slow Training:**
```python
# Enable packing
packing=True

# Reduce max_steps
max_steps=30  # instead of 60

# Use smaller dataset
dataset = dataset.select(range(500))  # first 500 samples
```

**3. Model Not Learning:**
```python
# Increase learning rate
learning_rate=5e-4  # for LoRA/CPT

# Increase LoRA rank
r=32  # instead of 16

# More training steps
max_steps=200
```

**4. CUDA Out of Memory:**
```python
# Clear cache
import torch
torch.cuda.empty_cache()

# Reduce batch size and accumulation
per_device_train_batch_size=1
gradient_accumulation_steps=8
```

---

## 💡 Tips for Success

### 1. Start Simple
- Use small models (SmolLM2-135M, Llama-3.2-1B)
- Use small datasets (1000 samples)
- Short training (60 steps)
- Then scale up

### 2. Monitor Training
- Watch loss curves
- Check GPU memory
- Time each run
- Save checkpoints

### 3. Test Thoroughly
- Before/after comparisons
- Multiple test cases
- Different prompts
- Edge cases

### 4. Document Everything
- Take screenshots
- Save outputs
- Record metrics
- Note observations

### 5. Understand Parameters
- Don't just copy values
- Experiment with changes
- Learn what each does
- Find optimal settings

---

## 🌟 Advanced Extensions

Want to go further? Try:

### 1. Combine Techniques
```
CPT → SFT → DPO
(Domain → Format → Align)
```

### 2. Export to Ollama
```python
model.save_pretrained_gguf("model", tokenizer, quantization_method="q4_k_m")
```

### 3. Multi-Stage Training
```python
# Stage 1: Continued Pretraining
# Stage 2: Instruction Finetuning
# Stage 3: DPO Alignment
```

### 4. Custom Datasets
- Create your own domain data
- Collect preference pairs
- Design reward functions
- Build specialized models

### 5. Larger Models
- Try Llama-3.1-8B
- Phi-3.5-mini
- Mistral-7B
- (Requires more GPU memory)

---

## 📚 Key Resources

### Official Documentation:
- [Unsloth Docs](https://docs.unsloth.ai/)
- [Unsloth GitHub](https://github.com/unslothai/unsloth)
- [Unsloth Notebooks](https://github.com/unslothai/notebooks)

### Tutorials:
- [Fine-tuning Guide](https://docs.unsloth.ai/get-started/fine-tuning-llms-guide)
- [RL Guide](https://docs.unsloth.ai/get-started/reinforcement-learning-rl-guide)
- [CPT Guide](https://docs.unsloth.ai/basics/continued-pretraining)

### Datasets:
- [Hugging Face Datasets](https://huggingface.co/datasets)
- [Alpaca Dataset](https://huggingface.co/datasets/yahma/alpaca-cleaned)
- [DPO Pairs](https://huggingface.co/datasets/Intel/orca_dpo_pairs)
- [Code Datasets](https://huggingface.co/datasets/code_search_net)

---

## 🎯 Grading Criteria (Reference)

Based on assignment requirements:

1. **Code Execution (40%):**
   - All notebooks run successfully
   - No errors
   - Proper outputs

2. **Video Quality (30%):**
   - Clear explanations
   - Detailed walkthrough
   - Shows input/output
   - Explains concepts

3. **Understanding (20%):**
   - Explains how it works
   - Dataset format understanding
   - Parameter knowledge
   - Use case awareness

4. **Completeness (10%):**
   - All 5 notebooks
   - All 5 videos
   - Proper documentation
   - Timely submission

---

## 🤝 Support

If you need help:

1. **Check notebook comments:** Detailed explanations in each cell
2. **Read summaries:** Each notebook has comprehensive summary section
3. **Review documentation:** Links to official resources
4. **Experiment:** Try different parameters and observe

---

## 📄 Summary

You now have **5 complete notebooks** covering:

✅ **Full Finetuning** - Train all parameters  
✅ **LoRA Finetuning** - Efficient adapter training  
✅ **DPO Alignment** - Learn human preferences  
✅ **GRPO Reasoning** - Self-improving reasoning models  
✅ **Continued Pretraining** - Learn new domains  

Each notebook is:
- ✅ Fully functional and ready to run
- ✅ Extensively documented with explanations
- ✅ Contains working examples
- ✅ Includes best practices
- ✅ Designed for video walkthrough

**Good luck with your assignment! 🚀**

---

## 📞 Contact

For questions about the notebooks or techniques:
- Review the detailed comments in each notebook
- Check the summary sections
- Refer to official Unsloth documentation

**Remember:** The best way to learn is by running the code, experimenting with parameters, and understanding what each technique does!
