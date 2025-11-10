# Virtual Environment Setup Guide

## 🎯 Purpose
This guide helps you set up a Python virtual environment to install all dependencies locally in your project folder rather than globally.

## 📋 Prerequisites
- Python 3.8 or higher installed
- pip (Python package manager)
- Git (for cloning unsloth)

## 🚀 Quick Start

### Windows (PowerShell)
```powershell
# Run the setup script
.\setup_venv.ps1

# If you get execution policy error, run:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
# Then try again
.\setup_venv.ps1
```

### Linux/Mac or Git Bash
```bash
# Make script executable
chmod +x setup_venv.sh

# Run the setup script
./setup_venv.sh
```

### Manual Setup (Any Platform)

#### Step 1: Create Virtual Environment
```bash
# Windows
python -m venv venv

# Linux/Mac
python3 -m venv venv
```

#### Step 2: Activate Virtual Environment

**Windows PowerShell:**
```powershell
.\venv\Scripts\Activate.ps1
```

**Windows Command Prompt:**
```cmd
venv\Scripts\activate.bat
```

**Linux/Mac/Git Bash:**
```bash
source venv/bin/activate
```

#### Step 3: Install Dependencies
```bash
# Upgrade pip first
python -m pip install --upgrade pip

# Install all requirements
pip install -r requirements.txt
```

## 📦 What Gets Installed

### Core Libraries:
- **PyTorch** - Deep learning framework
- **Unsloth** - Fast LLM training library
- **Transformers** - Hugging Face transformers
- **TRL** - Transformer Reinforcement Learning
- **PEFT** - Parameter-Efficient Fine-Tuning
- **Accelerate** - Distributed training
- **Datasets** - Dataset management

### Training Libraries:
- **bitsandbytes** - 4-bit/8-bit quantization
- **xformers** - Memory-efficient attention
- **triton** - GPU kernels

### Utilities:
- **Jupyter** - Notebook interface
- **NumPy, SciPy** - Scientific computing
- **SymPy** - Symbolic mathematics
- **HuggingFace Hub** - Model sharing

## 🔍 Verify Installation

After setup, verify everything works:

```python
# Activate venv first, then run Python
python

# In Python interpreter:
>>> import torch
>>> import unsloth
>>> import transformers
>>> print(torch.cuda.is_available())  # Should be True if GPU available
>>> print(torch.__version__)
>>> exit()
```

## 📂 Project Structure

After setup, your folder will look like:
```
unslothai/
├── venv/                          # Virtual environment (not committed)
│   ├── Scripts/ (Windows)
│   ├── bin/ (Linux/Mac)
│   └── Lib/
├── colab1_full_finetuning_smollm2.ipynb
├── colab2_lora_finetuning_smollm2.ipynb
├── colab3_dpo_reinforcement_learning.ipynb
├── colab4_grpo_reasoning_model.ipynb
├── colab5_continued_pretraining.ipynb
├── requirements.txt               # Dependencies list
├── setup_venv.ps1                 # Windows setup script
├── setup_venv.sh                  # Linux/Mac setup script
├── .gitignore                     # Git ignore file
├── README.md
├── QUICK_REFERENCE.md
└── TROUBLESHOOTING.md
```

## 🎓 Using with Jupyter Notebooks

### Option 1: Command Line
```bash
# Activate venv
.\venv\Scripts\Activate.ps1  # Windows
source venv/bin/activate     # Linux/Mac

# Start Jupyter
jupyter notebook

# Open your .ipynb files
```

### Option 2: VS Code
1. Open VS Code in the project folder
2. Install "Jupyter" extension
3. Open any .ipynb file
4. Click "Select Kernel" in top-right
5. Choose "Python Environments..."
6. Select `venv/Scripts/python.exe` (or `venv/bin/python`)

### Option 3: Google Colab (No venv needed)
- Upload .ipynb files to Google Colab
- Dependencies will be installed per notebook
- GPU is provided for free

## 🔄 Daily Workflow

### Start Working
```bash
# Activate venv
.\venv\Scripts\Activate.ps1  # Windows
source venv/bin/activate     # Linux/Mac

# You'll see (venv) in your prompt
```

### Stop Working
```bash
# Deactivate venv
deactivate
```

## 🛠️ Common Issues

### Issue: "Execution Policy" Error (Windows)
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Issue: Python Not Found
```bash
# Windows - try:
py -m venv venv

# Or specify full path:
C:\Python39\python.exe -m venv venv
```

### Issue: pip Install Fails
```bash
# Upgrade pip
python -m pip install --upgrade pip

# Try with no cache
pip install --no-cache-dir -r requirements.txt

# Or install individually
pip install torch transformers datasets
pip install unsloth
```

### Issue: CUDA Not Available
- Install CUDA-compatible PyTorch:
```bash
# Visit: https://pytorch.org/get-started/locally/
# Select your configuration and run the command
```

### Issue: Out of Disk Space
```bash
# Clean pip cache
pip cache purge

# Remove old packages
pip uninstall <package-name>
```

## 🌟 Benefits of Virtual Environment

✅ **Isolated Dependencies** - Each project has its own packages
✅ **No Global Pollution** - System Python stays clean
✅ **Version Control** - Specific versions per project
✅ **Easy Sharing** - Just share requirements.txt
✅ **No Admin Rights** - Install without sudo/admin
✅ **Multiple Projects** - Different versions for different projects

## 📝 Updating Dependencies

### Update Specific Package
```bash
pip install --upgrade package-name
```

### Update All Packages
```bash
pip list --outdated
pip install --upgrade package-name1 package-name2 ...
```

### Regenerate requirements.txt
```bash
pip freeze > requirements.txt
```

## 🗑️ Removing Virtual Environment

If you want to start fresh:

```bash
# Deactivate first
deactivate

# Remove the folder
# Windows
rmdir /s venv

# Linux/Mac
rm -rf venv

# Then recreate
python -m venv venv
```

## 🔐 For Google Colab Users

**Note:** Virtual environments are not needed for Google Colab. The notebooks install dependencies directly in cells:

```python
%%capture
!pip install unsloth
!pip install datasets transformers accelerate
```

Colab provides:
- ✅ Free GPU (T4)
- ✅ Pre-installed Python
- ✅ Temporary environment per session
- ✅ No local setup needed

## 📞 Need Help?

1. Check TROUBLESHOOTING.md
2. Ensure Python is in PATH
3. Try manual setup steps
4. Use Google Colab as alternative

## ✅ Quick Checklist

Before starting work:
- [ ] Virtual environment created
- [ ] Virtual environment activated (see `(venv)` in prompt)
- [ ] Dependencies installed
- [ ] Can import torch and unsloth
- [ ] GPU detected (if available)

**Happy Training! 🚀**
