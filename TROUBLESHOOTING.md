# Troubleshooting Guide

## 🔧 Common Issues and Solutions

### 0. **CRITICAL: Recursion Error with datasets 4.4.1** ⚠️

#### Problem: Recursion error when importing unsloth
```python
NotImplementedError: #### Unsloth: Using `datasets = 4.4.1` will cause recursion errors.
Please downgrade datasets to `datasets==4.3.0
```

**Solution:**
```python
# In Colab, run this in a cell BEFORE importing unsloth:
!pip install datasets==4.3.0
```

**Why it happens:**
- Unsloth is incompatible with datasets version 4.4.1+
- Version 4.3.0 is the last stable version that works

**Prevention:**
All notebooks have been updated with the correct version. If you see this error, simply downgrade datasets.

---

### 1. Installation Issues

#### Problem: `pip install unsloth` fails
```bash
Error: Could not find a version that satisfies the requirement unsloth
```

**Solution:**
```python
# Use the full installation command
%%capture
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps trl peft accelerate bitsandbytes
```

#### Problem: Import errors
```python
ModuleNotFoundError: No module named 'unsloth'
```

**Solution:**
```python
# Restart runtime after installation
# Runtime → Restart runtime
# Then re-run import cells
```

---

### 2. GPU/Memory Issues

#### Problem: CUDA Out of Memory
```
RuntimeError: CUDA out of memory. Tried to allocate XXX MiB
```

**Solutions:**

**Option 1: Reduce Batch Size**
```python
per_device_train_batch_size=1  # Instead of 2
gradient_accumulation_steps=8   # Increase to maintain effective batch size
```

**Option 2: Reduce Sequence Length**
```python
max_seq_length=1024  # Instead of 2048
```

**Option 3: Reduce LoRA Rank**
```python
r=8  # Instead of 16 or 32
```

**Option 4: Clear Cache**
```python
import torch
torch.cuda.empty_cache()
import gc
gc.collect()
```

**Option 5: Use Smaller Model**
```python
# Instead of Llama-3.2-1B, use:
model_name = "unsloth/SmolLM2-135M-Instruct"
```

#### Problem: GPU not detected
```
CUDA available: False
```

**Solution:**
```python
# In Google Colab:
# 1. Go to Runtime → Change runtime type
# 2. Select GPU (T4)
# 3. Click Save
# 4. Restart and re-run
```

---

### 3. Training Issues

#### Problem: Loss not decreasing
```
Step 50: Loss = 2.5
Step 100: Loss = 2.5
Step 150: Loss = 2.5
```

**Solutions:**

**Option 1: Increase Learning Rate**
```python
# For LoRA/CPT
learning_rate=5e-4  # Instead of 2e-4

# For DPO
learning_rate=1e-5  # Instead of 5e-6
```

**Option 2: Increase LoRA Rank**
```python
r=32  # Instead of 16
lora_alpha=32
```

**Option 3: More Training Steps**
```python
max_steps=200  # Instead of 60
num_train_epochs=5  # For CPT
```

**Option 4: Check Dataset**
```python
# Verify dataset is formatted correctly
print(dataset[0])
# Should show properly formatted text
```

#### Problem: Loss exploding (NaN)
```
Step 10: Loss = 1.5
Step 20: Loss = 5.2
Step 30: Loss = NaN
```

**Solutions:**

**Option 1: Lower Learning Rate**
```python
learning_rate=1e-5  # Much lower
```

**Option 2: Add Gradient Clipping**
```python
# In TrainingArguments
max_grad_norm=1.0
```

**Option 3: Check Data Quality**
```python
# Look for corrupted samples
for i, sample in enumerate(dataset):
    if len(sample['text']) == 0:
        print(f"Empty sample at index {i}")
```

#### Problem: Training very slow
```
Step 1/1000 - ETA: 5 hours
```

**Solutions:**

**Option 1: Enable Packing**
```python
packing=True  # For CPT especially
```

**Option 2: Reduce Steps**
```python
max_steps=60  # For quick demo
```

**Option 3: Use Smaller Dataset**
```python
dataset = dataset.select(range(1000))  # First 1000 samples
```

**Option 4: Increase Batch Size (if memory allows)**
```python
per_device_train_batch_size=4
gradient_accumulation_steps=2
```

---

### 4. Dataset Issues

#### Problem: Dataset not loading
```
FileNotFoundError: Dataset not found
```

**Solution:**
```python
# Check dataset name and split
dataset = load_dataset("yahma/alpaca-cleaned", split="train")

# If still fails, try with streaming
dataset = load_dataset("yahma/alpaca-cleaned", split="train", streaming=True)
dataset = dataset.take(1000)  # Take first 1000
```

#### Problem: Wrong dataset format
```
KeyError: 'text'
```

**Solution:**
```python
# Check what columns exist
print(dataset.column_names)

# Rename or map to correct format
if 'text' not in dataset.column_names:
    dataset = dataset.map(your_formatting_function, batched=True)
```

#### Problem: Dataset too large
```
Memory error while loading dataset
```

**Solution:**
```python
# Load only subset
dataset = load_dataset("dataset_name", split="train[:1000]")

# Or use streaming
dataset = load_dataset("dataset_name", split="train", streaming=True)
```

---

### 5. Model Issues

#### Problem: Model not generating
```
# Generate returns empty or repeating text
```

**Solution:**

**Option 1: Check Inference Mode**
```python
FastLanguageModel.for_inference(model)
```

**Option 2: Adjust Generation Parameters**
```python
outputs = model.generate(
    **inputs,
    max_new_tokens=150,  # Increase
    temperature=0.7,     # Adjust (0.1-1.0)
    top_p=0.9,
    do_sample=True,      # Enable sampling
    repetition_penalty=1.2  # Prevent repetition
)
```

**Option 3: Check Template**
```python
# Make sure prompt follows model's expected format
messages = [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
```

#### Problem: Model outputs gibberish
```
# After training, model produces nonsense
```

**Solution:**

**Check 1: Training worked?**
```python
# Look at final loss
# Should be < 1.0 for good results
```

**Check 2: Too high learning rate?**
```python
# Reduce learning rate and retrain
learning_rate=1e-5
```

**Check 3: Overfitting?**
```python
# Reduce training steps
max_steps=30
```

---

### 6. DPO-Specific Issues

#### Problem: DPOTrainer import fails
```
ImportError: cannot import name 'DPOTrainer'
```

**Solution:**
```python
# Install latest TRL
!pip install --upgrade trl

# Import after patching
from unsloth import PatchDPOTrainer
PatchDPOTrainer()
from trl import DPOTrainer, DPOConfig
```

#### Problem: DPO dataset format wrong
```
KeyError: 'chosen' or 'rejected'
```

**Solution:**
```python
# Verify dataset has required fields
print(dataset[0].keys())
# Should have: 'prompt', 'chosen', 'rejected'

# Format properly
def format_dpo(examples):
    return {
        "prompt": [...],
        "chosen": [...],
        "rejected": [...]
    }

dataset = dataset.map(format_dpo, batched=True)
```

---

### 7. GRPO-Specific Issues

#### Problem: GRPO not available
```
ImportError: cannot import name 'GRPOTrainer'
```

**Solution:**
```python
# GRPO is experimental - check TRL version
!pip install git+https://github.com/huggingface/trl.git

# Or use the demonstration code in Colab 4
# which shows the concept without full GRPO
```

#### Problem: Reward function not working
```
# Rewards always 0.0
```

**Solution:**
```python
# Debug reward function
def reward_function(prompts, responses, answers):
    for resp, ans in zip(responses, answers):
        predicted = extract_answer(resp)
        print(f"Predicted: {predicted}, Answer: {ans}")
        # Check if extraction works
    return rewards

# Test extraction function separately
test_text = "The answer is 42"
print(extract_answer(test_text))  # Should print "42"
```

---

### 8. Saving/Loading Issues

#### Problem: Can't save model
```
PermissionError: [Errno 13] Permission denied
```

**Solution:**
```python
# Use absolute path
model.save_pretrained("/content/my_model")

# Or use Google Drive (mount first)
from google.colab import drive
drive.mount('/content/drive')
model.save_pretrained("/content/drive/MyDrive/my_model")
```

#### Problem: Can't load saved model
```
OSError: Model not found
```

**Solution:**
```python
# Check path exists
import os
print(os.listdir("/content/my_model"))

# Load with correct path
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/content/my_model",  # Full path
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True
)
```

---

### 9. Runtime Issues

#### Problem: Colab disconnects
```
Runtime disconnected. Reconnect?
```

**Solutions:**

**Prevention:**
```javascript
// Keep session alive (run in browser console)
function ClickConnect(){
  console.log("Working");
  document.querySelector("colab-toolbar-button#connect").click()
}
setInterval(ClickConnect, 60000)
```

**Recovery:**
```python
# Save checkpoints regularly
# In TrainingArguments:
save_steps=50,
save_total_limit=2,
output_dir="./checkpoints"

# Resume from checkpoint
trainer.train(resume_from_checkpoint="./checkpoints/checkpoint-50")
```

#### Problem: Runtime out of resources
```
Your session crashed after using all available RAM
```

**Solution:**
```python
# Reduce memory usage:
1. Smaller batch size
2. Shorter sequences
3. Smaller dataset
4. Lower LoRA rank
5. More gradient accumulation

# Example:
per_device_train_batch_size=1
max_seq_length=512
r=8
gradient_accumulation_steps=16
```

---

### 10. Video Recording Issues

#### Problem: Screen recording laggy

**Solution:**
- Record in segments
- Use OBS Studio or similar
- Record at 720p instead of 1080p
- Close other applications

#### Problem: File too large

**Solution:**
```bash
# Compress video using ffmpeg
ffmpeg -i input.mp4 -vcodec h264 -acodec aac -b:v 1000k output.mp4

# Or upload to YouTube as unlisted
```

---

## 🚨 Emergency Fixes

### Quick Reset
```python
# If everything is broken:
# 1. Runtime → Restart runtime
# 2. Clear all outputs
# 3. Run cells from top

# Or factory reset:
# Runtime → Factory reset runtime
```

### Start Fresh
```python
# Create new notebook
# Copy cells one by one
# Test each cell individually
```

---

## 📞 Getting Help

### Before Asking:
1. ✅ Check this troubleshooting guide
2. ✅ Read error message carefully
3. ✅ Check notebook comments
4. ✅ Review README.md
5. ✅ Try suggested solutions

### Where to Look:
- 📖 [Unsloth Docs](https://docs.unsloth.ai/)
- 💬 [Unsloth Discord](https://discord.gg/unsloth)
- 🐛 [GitHub Issues](https://github.com/unslothai/unsloth/issues)
- 📚 [Notebook Examples](https://github.com/unslothai/notebooks)

---

## ✅ Verification Checklist

Before recording video, verify:
- [ ] GPU is enabled (T4 or better)
- [ ] All imports work
- [ ] Dataset loads correctly
- [ ] Model loads without errors
- [ ] Training runs for at least 10 steps
- [ ] Loss is decreasing (or stable for DPO)
- [ ] Inference generates text
- [ ] Can save model
- [ ] No memory errors

---

## 🎯 Quick Fixes Summary

| Issue | Quick Fix |
|-------|-----------|
| Out of memory | Reduce batch size to 1 |
| Too slow | Enable packing, reduce steps |
| Not learning | Increase LR, increase r |
| Loss NaN | Reduce LR, add grad clipping |
| Import error | Upgrade packages, restart runtime |
| Dataset error | Check format, use .select() for subset |
| GPU not found | Change runtime type to GPU |
| Can't save | Use absolute path |

---

## 💡 Pro Tips

### Tip 1: Test with Minimal Settings First
```python
# Start with:
max_steps=10
per_device_train_batch_size=1
dataset = dataset.select(range(100))

# Once working, scale up
```

### Tip 2: Save Incrementally
```python
# Don't wait until end
save_steps=20
save_total_limit=3
```

### Tip 3: Monitor Resources
```python
# Check GPU usage
!nvidia-smi

# Check disk space
!df -h
```

### Tip 4: Use Colab Pro (Optional)
- Better GPU (A100, V100)
- More RAM
- Longer runtime
- Faster training

---

**Remember: Most issues are solved by reducing memory usage or checking dataset format! 🔧**
