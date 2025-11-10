# Quick Reference Guide - Key Concepts

## 🎯 Training Methods Comparison

### 1. Full Finetuning (Colab 1)
```python
# Key Setting: r=0 (no LoRA)
model = FastLanguageModel.get_peft_model(
    model,
    r=0,  # ← This makes it FULL finetuning
    use_gradient_checkpointing="unsloth"
)
```
**Updates:** ALL 135M parameters  
**Memory:** High  
**Speed:** Slower  
**Use:** Critical tasks requiring maximum performance  

---

### 2. LoRA Finetuning (Colab 2)
```python
# Key Setting: r>0 (enables LoRA)
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # ← This enables LoRA adapters
    lora_alpha=16,
    lora_dropout=0
)
```
**Updates:** ~1-2M parameters (1-2%)  
**Memory:** Low  
**Speed:** Fast  
**Use:** Most general tasks  

---

### 3. DPO Alignment (Colab 3)
```python
# Key: Uses DPOTrainer with preference pairs
from trl import DPOTrainer, DPOConfig

trainer = DPOTrainer(
    model=model,
    args=DPOConfig(
        beta=0.1,  # ← DPO-specific parameter
        learning_rate=5e-6  # Lower than SFT
    ),
    # Dataset needs: prompt + chosen + rejected
)
```
**Dataset:** Prompt + Chosen + Rejected  
**Purpose:** Learn human preferences  
**Use:** Alignment after SFT  

---

### 4. GRPO Reasoning (Colab 4)
```python
# Key: Multiple generations + reward function
from trl import GRPOTrainer, GRPOConfig

def reward_function(responses, answers):
    return [1.0 if correct else 0.0]

trainer = GRPOTrainer(
    model=model,
    args=GRPOConfig(
        num_generations=4,  # ← Generate multiple solutions
    ),
    reward_fn=reward_function  # ← Evaluate correctness
)
```
**Dataset:** Prompts only  
**Purpose:** Self-improvement, reasoning  
**Use:** Math, logic, problem-solving  

---

### 5. Continued Pretraining (Colab 5)
```python
# Key: Raw text, higher LR, no templates
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,  # ← Raw text only!
    dataset_text_field="text",
    packing=True,  # ← Important for CPT
    args=TrainingArguments(
        learning_rate=1e-4,  # ← Higher than SFT
        num_train_epochs=3   # ← More epochs
    )
)
```
**Dataset:** Raw domain text  
**Purpose:** Learn new knowledge/domain  
**Use:** New languages, domains, knowledge  

---

## 📊 Dataset Format Reference

### SFT (Colab 1-2)
```json
{
  "instruction": "What is machine learning?",
  "input": "",
  "output": "Machine learning is..."
}
```

### DPO (Colab 3)
```json
{
  "prompt": "Explain AI",
  "chosen": "Detailed, helpful response...",
  "rejected": "Short, unhelpful response"
}
```

### GRPO (Colab 4)
```json
{
  "query": "What is 15 × 23?",
  "answer": "345"
}
```

### CPT (Colab 5)
```json
{
  "text": "Raw domain text without any template..."
}
```

---

## ⚙️ Key Parameter Differences

| Parameter | Full FT | LoRA | DPO | GRPO | CPT |
|-----------|---------|------|-----|------|-----|
| **r** | 0 | 16-64 | 16-32 | 32-64 | 32-64 |
| **Learning Rate** | 2e-4 | 2e-4 | 5e-6 | 1e-5 | 1e-4 |
| **Epochs** | 1 | 1 | 1 | 1 | 3-10 |
| **Batch Size** | 2 | 2 | 2 | 1 | 2 |
| **Packing** | False | False | N/A | N/A | True |

---

## 🎓 When to Use Each Method

### Use Full Finetuning when:
- ✅ Maximum performance required
- ✅ Critical applications
- ✅ Have sufficient GPU memory
- ✅ Training time not critical

### Use LoRA when:
- ✅ Limited GPU memory
- ✅ Need fast training
- ✅ Good performance sufficient
- ✅ Want small adapters to share

### Use DPO when:
- ✅ After SFT training
- ✅ Have preference data
- ✅ Need alignment
- ✅ Improve response quality

### Use GRPO when:
- ✅ Training reasoning models
- ✅ Have automatic evaluation
- ✅ Math/logic tasks
- ✅ Self-improvement needed

### Use CPT when:
- ✅ Teaching new domain
- ✅ Adding new language
- ✅ Have raw domain text
- ✅ Need deep expertise

---

## 💡 Quick Tips

### Memory Issues?
```python
# Reduce these:
per_device_train_batch_size=1
max_seq_length=1024
r=8  # for LoRA
```

### Training Too Slow?
```python
# Try these:
packing=True
max_steps=30  # reduce steps
gradient_accumulation_steps=8
```

### Not Learning?
```python
# Increase these:
learning_rate=5e-4
r=32  # for LoRA
num_train_epochs=3
```

---

## 🚀 Typical Pipeline

```
1. Continued Pretraining (if new domain)
   ↓
2. Instruction Finetuning (SFT - LoRA or Full)
   ↓
3. DPO Alignment (optional)
   ↓
4. GRPO (for reasoning tasks)
   ↓
5. Production Model!
```

---

## 📈 Expected Training Times (T4 GPU)

- **Full Finetuning:** 10-15 minutes (60 steps)
- **LoRA:** 5-10 minutes (60 steps)
- **DPO:** 10-15 minutes (60 steps)
- **GRPO:** 15-20 minutes (100 steps)
- **CPT:** 15-20 minutes (3 epochs)

---

## 🔍 How to Verify Success

### Full Finetuning:
- Check: All parameters updated
- Test: Better task performance
- Compare: Memory usage vs LoRA

### LoRA:
- Check: Small adapter size (~10-50MB)
- Test: Similar performance to full
- Verify: Can merge or separate adapters

### DPO:
- Check: Loss decreasing
- Test: Prefers better responses
- Compare: Before/after quality

### GRPO:
- Check: Reward increasing
- Test: Correct answers
- Verify: Reasoning improved

### CPT:
- Check: Perplexity on domain text
- Test: Domain knowledge
- Compare: Before/after on domain tasks

---

## 📝 Video Recording Checklist

For each notebook:
- [ ] Show notebook title/overview
- [ ] Explain what method does
- [ ] Show dataset format
- [ ] Walk through key parameters
- [ ] Run training (can speed up video)
- [ ] Show training metrics
- [ ] Test with inference examples
- [ ] Compare before/after
- [ ] Summarize key learnings

Recommended structure:
1. Introduction (2 min)
2. Setup (2 min)
3. Dataset (3 min)
4. Configuration (5 min)
5. Training (5 min)
6. Results (5 min)
7. Testing (5 min)
8. Summary (3 min)

**Total: ~30 minutes per notebook**

---

## 🎯 Assignment Submission Format

```
submission/
├── colab1_full_finetuning_smollm2.ipynb
├── colab1_video.mp4 (or YouTube link)
├── colab2_lora_finetuning_smollm2.ipynb
├── colab2_video.mp4
├── colab3_dpo_reinforcement_learning.ipynb
├── colab3_video.mp4
├── colab4_grpo_reasoning_model.ipynb
├── colab4_video.mp4
├── colab5_continued_pretraining.ipynb
├── colab5_video.mp4
└── README.md (this file)
```

---

## ✅ Final Checklist

Before submission:
- [ ] All 5 notebooks run successfully
- [ ] All 5 videos recorded and uploaded
- [ ] Each video shows full walkthrough
- [ ] Code is well-commented
- [ ] Results are clearly shown
- [ ] Explanations are clear
- [ ] No errors in notebooks
- [ ] Videos are clear and audible

---

## 🌟 Bonus Points Ideas

Want to stand out? Try:
1. Compare all 5 methods on same task
2. Create visualizations of training curves
3. Export to GGUF and use with Ollama
4. Add your own custom dataset
5. Show real-world use case
6. Include ablation studies
7. Demonstrate multi-stage training

---

**Good luck! You have everything you need to complete this assignment successfully! 🚀**
