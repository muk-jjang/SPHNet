# SPHNet Bond Stretch Experiment - Documentation Index

## 📚 Quick Navigation

### 🚀 **[QUICK_START.md](QUICK_START.md)** - Start Here!
**For users who want to get started immediately**
- One-command setup
- Common use cases
- Minimal examples
- Quick troubleshooting

### 📖 **[EXPERIMENT_README.md](EXPERIMENT_README.md)** - Complete Guide
**For comprehensive understanding of the experiment**
- Full experiment workflow
- Detailed usage instructions
- Data format specifications
- Step-by-step examples

### 🔧 **[SCRIPTS_README.md](SCRIPTS_README.md)** - Bash Scripts Reference
**For users running automated predictions**
- Script descriptions
- All command-line options
- Advanced usage patterns
- Error handling

### 📊 **[WORKFLOW.md](WORKFLOW.md)** - Visual Diagrams
**For understanding the complete workflow**
- Architecture diagrams
- Data flow charts
- File structure
- Execution flow

### 📝 **[SUMMARY.md](SUMMARY.md)** - Technical Summary
**For developers and technical details**
- Created/modified files
- Implementation details
- Code architecture
- API references

### ⚙️ **[CONFIG_SETUP_GUIDE.md](CONFIG_SETUP_GUIDE.md)** - Checkpoint Configuration
**For setting up per-molecule checkpoints**
- Why per-molecule checkpoints matter
- How to configure `exp_config_sphnet.yaml`
- Validation and troubleshooting
- **START HERE if you have multiple checkpoints**

---

## 📁 File Organization

### Python Scripts
```
SPHNet/
└── exp_bond-stretch_model.py          # SPHNet prediction script

QHFlow-mlff/src/electronic_structure_exp/
├── exp_bond-stretch_dft.py             # DFT calculations
├── exp_bond-stretch_model.py           # QHFlow-v2 predictions
├── exp_bond-stretch_plot.py            # Two-method plots
└── exp_bond-stretch_plot_three_methods.py  # Three-method plots
```

### Bash Scripts (SPHNet/)
```
├── run_sphnet_predictions.sh           # Run all SPHNet predictions
├── run_three_method_plots.sh           # Generate all comparison plots
└── run_all_experiments.sh              # Master script (predictions + plots)
```

### Documentation (SPHNet/)
```
├── README_INDEX.md                     # This file
├── QUICK_START.md                      # Quick start guide
├── CONFIG_SETUP_GUIDE.md               # Checkpoint configuration guide
├── EXPERIMENT_README.md                # Complete usage guide
├── SCRIPTS_README.md                   # Bash scripts documentation
├── WORKFLOW.md                         # Visual workflow diagrams
├── SUMMARY.md                          # Technical summary
└── UPDATE_PER_MOLECULE_CKPT.md         # Per-molecule checkpoint update notes
```

### Configuration (SPHNet/)
```
└── exp_config_sphnet.yaml              # Per-molecule checkpoint paths
```

---

## 🎯 Choose Your Path

### Path 1: "Just Make It Work" → [QUICK_START.md](QUICK_START.md) + [CONFIG_SETUP_GUIDE.md](CONFIG_SETUP_GUIDE.md)
- You have checkpoints for each molecule
- Want to set up and run immediately
- Need to configure checkpoint paths first

### Path 2: "Understand Everything" → [EXPERIMENT_README.md](EXPERIMENT_README.md)
- Want to understand the experiment design
- Need to know data formats
- Plan to modify or extend the code

### Path 3: "Automate My Workflow" → [SCRIPTS_README.md](SCRIPTS_README.md)
- Need to run many molecules efficiently
- Want to customize script behavior
- Need detailed option documentation

### Path 4: "Visual Learner" → [WORKFLOW.md](WORKFLOW.md)
- Prefer diagrams to text
- Want to see data flow
- Need architecture overview

### Path 5: "Developer/Technical" → [SUMMARY.md](SUMMARY.md)
- Modifying the code
- Understanding implementation
- Comparing with other methods

---

## 🔍 Find What You Need

### "How do I run predictions?"
- **Setup checkpoints first**: [CONFIG_SETUP_GUIDE.md](CONFIG_SETUP_GUIDE.md)
- **Quick start**: [QUICK_START.md § Use Cases](QUICK_START.md#common-use-cases)
- **Detailed**: [EXPERIMENT_README.md § Usage](EXPERIMENT_README.md#usage)
- **Automated**: [SCRIPTS_README.md § run_sphnet_predictions.sh](SCRIPTS_README.md#1-run_sphnet_predictionssh---run-sphnet-predictions)

### "What files will be created?"
- **Quick**: [QUICK_START.md § File Naming](QUICK_START.md#file-naming-convention)
- **Detailed**: [EXPERIMENT_README.md § Output Files](EXPERIMENT_README.md#output-files)
- **Visual**: [WORKFLOW.md § File Flow](WORKFLOW.md#file-flow-diagram)

### "How do I compare all three methods?"
- **Quick**: [QUICK_START.md § TL;DR](QUICK_START.md#tldr---run-everything)
- **Scripts**: [SCRIPTS_README.md § Master Script](SCRIPTS_README.md#3-run_all_experimentssh---complete-workflow)
- **Visual**: [WORKFLOW.md § Automated Workflow](WORKFLOW.md#automated-workflow-with-bash-scripts)

### "What data format is used?"
- **Detailed**: [EXPERIMENT_README.md § Data Format](EXPERIMENT_README.md#data-format)
- **Technical**: [SUMMARY.md § Data Format Compatibility](SUMMARY.md#data-format-compatibility)
- **Visual**: [WORKFLOW.md § Data Structure](WORKFLOW.md#data-structure-inside-pt-files)

### "Something is not working"
- **Quick fixes**: [QUICK_START.md § Troubleshooting](QUICK_START.md#troubleshooting)
- **Detailed help**: [EXPERIMENT_README.md § Troubleshooting](EXPERIMENT_README.md#troubleshooting)
- **Visual flow**: [WORKFLOW.md § Troubleshooting Flow](WORKFLOW.md#troubleshooting-flow)

### "What are the command-line options?"
- **All options**: [SCRIPTS_README.md](SCRIPTS_README.md)
- **Examples**: [QUICK_START.md § Common Use Cases](QUICK_START.md#common-use-cases)
- **Reference**: [WORKFLOW.md § Quick Command Reference](WORKFLOW.md#quick-command-reference)

---

## 📖 Documentation Details

| Document | Pages | Best For | Reading Time |
|----------|-------|----------|--------------|
| **QUICK_START.md** | Short | Beginners, quick setup | 5 min |
| **EXPERIMENT_README.md** | Long | Complete understanding | 20 min |
| **SCRIPTS_README.md** | Medium | Script automation | 15 min |
| **WORKFLOW.md** | Visual | Visual learners | 10 min |
| **SUMMARY.md** | Technical | Developers | 10 min |

---

## 🆘 Getting Help

1. **Start with**: [QUICK_START.md](QUICK_START.md)
2. **Still stuck?** Check [EXPERIMENT_README.md § Troubleshooting](EXPERIMENT_README.md#troubleshooting)
3. **Need script help?** See [SCRIPTS_README.md](SCRIPTS_README.md)
4. **Want to understand workflow?** Read [WORKFLOW.md](WORKFLOW.md)
5. **Technical issues?** Consult [SUMMARY.md](SUMMARY.md)

---

## 🎓 Learning Path

### Beginner
1. Read [QUICK_START.md](QUICK_START.md)
2. Run the one-line command
3. View your results

### Intermediate
1. Read [EXPERIMENT_README.md](EXPERIMENT_README.md)
2. Understand data formats
3. Customize your workflow

### Advanced
1. Read [SCRIPTS_README.md](SCRIPTS_README.md)
2. Study [WORKFLOW.md](WORKFLOW.md)
3. Review [SUMMARY.md](SUMMARY.md)
4. Modify scripts for your needs

---

## 🔗 External Dependencies

### Required
- Python 3.8+
- PyTorch
- PyTorch Geometric
- PySCF (for DFT calculations)
- NumPy, Matplotlib

### Optional
- CUDA (for GPU acceleration)
- gpu4pyscf (for faster DFT)

See [EXPERIMENT_README.md § Prerequisites](EXPERIMENT_README.md#prerequisites) for detailed setup.

---

## 📞 Quick Links

- **Main experiment guide**: [EXPERIMENT_README.md](EXPERIMENT_README.md)
- **Get started now**: [QUICK_START.md](QUICK_START.md)
- **Script documentation**: [SCRIPTS_README.md](SCRIPTS_README.md)
- **Workflow diagrams**: [WORKFLOW.md](WORKFLOW.md)
- **Technical details**: [SUMMARY.md](SUMMARY.md)

---

## 💡 Tips

- Use `--dry-run` to preview commands before executing
- Start with `QUICK_START.md` for immediate results
- Use bash scripts for batch processing
- Check `WORKFLOW.md` for visual understanding
- Consult `SCRIPTS_README.md` for advanced options

---

**Last Updated**: 2026-01-20

This documentation covers SPHNet bond stretch Hamiltonian prediction experiments for comparing DFT, QHFlow-v2, and SPHNet methods.
