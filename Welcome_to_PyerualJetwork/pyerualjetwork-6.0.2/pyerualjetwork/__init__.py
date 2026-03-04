""" 
PyerualJetwork
==============
PyereualJetwork is a large wide GPU-accelerated deep learning library in Python designed for professionals and researchers.
It features PLAN, MLP Deep Learning and PTNN training, as well as ENE (Eugenic NeuroEvolution) for genetic optimization, 
which can also be applied to genetic algorithms or Reinforcement Learning (RL) problems. And >6 version have classic deep learning optimizations. 
The library includes functions for data pre-processing, visualizations, model saving and loading, prediction and evaluation, 
training, and both detailed and simplified memory management.
PyerualJetwork Main Modules:
----------------------------
- nn
- ene
- model_ops
CPU Main Modules:
---------------------------
- cpu.data_ops
GPU Main Modules:
---------------------------
- cuda.data_ops
Memory Module:
--------------
- memory_ops
Issue Solver Module:
--------------
- issue_solver
Examples: https://github.com/HCB06/PyerualJetwork/tree/main/Welcome_to_PyerualJetwork/ExampleCodes
PyerualJetwork document: https://github.com/HCB06/PyerualJetwork/blob/main/Welcome_to_PyerualJetwork/PYERUALJETWORK_USER_MANUEL_AND_LEGAL_INFORMATION(EN).pdf
- Creator: Hasan Can Beydili
- YouTube: https://www.youtube.com/@HasanCanBeydili
- Linkedin: https://www.linkedin.com/in/hasan-can-beydili-77a1b9270/
- Instagram: https://www.instagram.com/canbeydili
- Contact: tchasancan@gmail.com
"""

__version__ = "6.0.2"
__update__ = """* Changes: https://github.com/HCB06/PyerualJetwork/blob/main/CHANGES
* PyerualJetwork Homepage: https://github.com/HCB06/PyerualJetwork/tree/main
* PyerualJetwork document: https://github.com/HCB06/PyerualJetwork/blob/main/Welcome_to_PyerualJetwork/PYERUALJETWORK_USER_MANUEL_AND_LEGAL_INFORMATION(EN).pdf
* YouTube tutorials: https://www.youtube.com/@HasanCanBeydili"""

def print_version(version):
    print(f"PyerualJetwork Version {version}\n")

def print_update_notes(update):
    print(f"Notes:\n{update}")

#print_version(__version__)
#print_update_notes(__update__)

import warnings
import importlib
import importlib.metadata

warnings.filterwarnings('ignore')

required_modules = {
    "scipy": "1.13.1",
    "tqdm": "4.66.4",
    "pandas": "2.2.2",
    "networkx": "3.3",
    "seaborn": "0.13.2",
    "numpy": "1.26.4",
    "matplotlib": "3.9.0",
    "colorama": "0.4.6",
    "psutil": "6.1.1"
}

missing_modules = []
for module, version in required_modules.items():
    try:
        installed = importlib.metadata.version(module)
        if installed != version:
            missing_modules.append(f"{module}=={version}")
    except importlib.metadata.PackageNotFoundError:
        missing_modules.append(f"{module}=={version}")

if missing_modules:
    raise ImportError(
        f"Missing modules detected: {', '.join(missing_modules)}\n"
        "Please run the following command to install the missing packages:\n\n"
        f"    pip install {' '.join(missing_modules)}\n\n"
        "NOTE: needed numpy version --> numpy==1.26.4 and cupy is OPTIONAL for CUDA GPU training option. "
        "If you have cuda toolkit version 12 you need to install with pip --> 'pip install cupy-cuda12x'. "
        "Learn your CUDA version: type into your terminal --> 'nvidia-smi'.\n"
        "For more information, visit the PyerualJetwork GitHub README.md file:\n"
        "https://github.com/HCB06/PyerualJetwork/blob/main/README.md"
    )