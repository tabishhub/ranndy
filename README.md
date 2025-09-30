# Randomized Neural Networks for Transfer Operators (RaNNDy)

This repository provides code and experiments for **RaNNDy**: a randomized neural network approach for learning of transfer operators (Koopman, forward-backward, etc.) and their spectral decompositions.  

The method is described in the paper:  
**“How deep is your network? Deep vs. shallow learning of transfer operators”**  
by Mohammad Tabish, Benedict Leimkuhler, and Stefan Klus (2025).  
👉 [Read the paper here](https://arxiv.org/abs/2509.19930)

---

## Features

- Randomized neural networks with fixed hidden layers, trained via closed-form for the output layer 
- Computation of spectral decompositions (eigenvalues, eigenfunctions)  
- Uncertainty quantification via ensembles  
- Jupyter notebooks demonstrating experiments 


---

## ⚙️ Installation & Dependencies

### Prerequisites
- Python 3.8+  
- Jupyter Notebook or Jupyter Lab  

### Installation

```bash
# Clone the repository
git clone https://github.com/tabishhub/ranndy.git
cd ranndy

# Create and activate a virtual environment (optional)
python3 -m venv venv
source venv/bin/activate  

# Install dependencies
pip install -r requirements.txt

```
#### Please make sure to install 'deeptime' and 'd3s' from the following links and set proper paths in the notebooks before running them.
- [Deeptime](https://github.com/deeptime-ml/deeptime) <br>
- [d3s](https://github.com/sklus/d3s)
- Protein molecules simulation data can be requested here [D.E. Shaw Research](https://www.deshawresearch.com/)

### Reference
```bibtex
@article{TLK2025,
  title   = {How deep is your network? Deep vs. shallow learning of transfer operators},
  author  = {Tabish, Mohammad and Leimkuhler, Benedict and Klus, Stefan},
  year    = {2025},
  url  = {[arXiv:2509.19930](https://arxiv.org/abs/2509.19930)}
}
```

---
