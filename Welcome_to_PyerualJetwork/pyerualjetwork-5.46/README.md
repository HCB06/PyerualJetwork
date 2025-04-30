# PyerualJetwork [![Socket Badge](https://socket.dev/api/badge/pypi/package/pyerualjetwork/4.0.6?artifact_id=tar-gz)](https://socket.dev/pypi/package/pyerualjetwork/overview/4.0.6/tar-gz) [![CodeFactor](https://www.codefactor.io/repository/github/hcb06/pyerualjetwork/badge)](https://www.codefactor.io/repository/github/hcb06/pyerualjetwork) [![PyPI Downloads](https://static.pepy.tech/badge/pyerualjetwork)](https://pepy.tech/projects/pyerualjetwork) + [![PyPI Downloads](https://static.pepy.tech/badge/anaplan)](https://pepy.tech/projects/anaplan)


[![PyPI Downloads](https://static.pepy.tech/badge/pyerualjetwork/month)](https://pepy.tech/projects/pyerualjetwork) [![PyPI Downloads](https://static.pepy.tech/badge/pyerualjetwork/week)](https://pepy.tech/projects/pyerualjetwork) [![PyPI version](https://img.shields.io/pypi/v/pyerualjetwork.svg)](https://pypi.org/project/pyerualjetwork/)

Note: anaplan old name of pyerualjetwork

![PyerualJetwork](https://github.com/HCB06/PyerualJetwork/blob/main/Media/pyerualjetwork_with_name.png)<br><br><br>

Libraries.io Page: https://libraries.io/pypi/pyerualjetwork

PyPi Page: https://pypi.org/project/pyerualjetwork/

GitHub Page: https://github.com/HCB06/PyerualJetwork

YouTube Tutorials: https://www.youtube.com/watch?v=6wMQstZ00is&list=PLNgNWpM7HbsBpCx2VTJ4SK9wcPyse-EHw

	installation: 
	'pip install pyerualjetwork'
	
	package modules:
	'from pyerualjetwork import nn, ene, model_ops, memory_ops'
	'from pyerualjetwork.cpu data_ops'
	'from pyerualjetwork.cuda data_ops'

	please read docstrings.
	
	PyerualJetwork has Issue Solver. This operation provides users ready-to-use functions to identify potential issues
 	caused by version incompatibilities in major updates, ensuring users are not affected by such problems. 
  	PyereualJetwork aims to offer a seamless experience for its users.

	from pyerualjetwork import issue_solver

      Optimized for Visual Studio Code
      
      requires=[
 	    'scipy==1.13.1',
	    'tqdm==4.66.4',
	    'pandas==2.2.2',
	    'networkx==3.3',
	    'seaborn==0.13.2',
	    'numpy==1.26.4',
	    'matplotlib==3.9.0',
	    'colorama==0.4.6',
        'cupy-cuda12x',
	    'psutil==6.1.1'
        ]

     matplotlib, networkx, seaborn (optional).
          
##############################

ABOUT PYERUALJETWORK:

PyereualJetwork is a large wide GPU-accelerated machine learning library in Python designed for professionals and researchers.
It features PLAN, MLP Deep Learning and PTNN training, as well as ENE (Eugenic NeuroEvolution) for genetic optimization, 
which can also be applied to genetic algorithms or Reinforcement Learning (RL) problems. 
The library includes functions for data pre-processing, visualizations, model saving and loading, prediction and evaluation, 
training, and both detailed and simplified memory management. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4862342. (THIS ARTICLE IS FIRST VERSION OF PLAN.) MODERN VERSION OF PLAN: https://github.com/HCB06/PyerualJetwork/blob/main/Welcome_to_PLAN/PLAN.pdf
Both the PLAN algorithm ENE algorithm and the PyerualJetwork library were created by Author, and all rights are reserved by Author.
PyerualJetwork is free to use for commercial business and individual users.

PyerualJetwork ready for both eager execution(like PyTorch) and static graph(like Tensorflow) concepts because PyerualJetwork using only functions.
For example:

plan_fit function only fits given training data(suitable for dynamic graph) but learn function learns and optimize entire architecture(suitable for static graph). Or more deeper eager executions PyerualJetwork have: cross_over function, mutation function, list of activation functions, loss functions. You can create your unique model architecture. Move your data to GPU or CPU or manage how much should in GPU, Its all up to you.
<br><br>

PyerualJetworket includes PLAN, MLP, PTNN & ENE.<br>

PLAN VISION:<br>

![PLAN VISION](https://github.com/HCB06/PyerualJetwork/blob/main/Media/PlanVision.jpg)

You can create artificial intelligence models that perform computer vision tasks using the neu module:<br>

![AUTONOMOUS](https://github.com/HCB06/PyerualJetwork/blob/main/Media/autonomous.gif)<br><br><br>
![XRAY](https://github.com/HCB06/PyerualJetwork/blob/main/Media/chest_xray.png)<br><br><br>
![GENDER](https://github.com/HCB06/PyerualJetwork/blob/main/Media/gender_classification.png)<br><br><br>

NLPlan:<br>

![NLPLAN](https://github.com/HCB06/PyerualJetwork/blob/main/Media/NLPlan.jpg)<br>

You can create artificial intelligence models that perform natural language processing tasks using the neu module:

![PLAN VISION](https://github.com/HCB06/PyerualJetwork/blob/main/Media/NLP.gif)

ENE:<br>

You can create artificial intelligence models that perform reinforcement learning tasks and genetic optimization tasks using the ene module:

![ENE](https://github.com/HCB06/PyerualJetwork/blob/main/Media/PLANEAT_1.gif)<br>
![ENE](https://github.com/HCB06/PyerualJetwork/blob/main/Media/PLANEAT_2.gif)<br>
![ENE](https://github.com/HCB06/PyerualJetwork/blob/main/Media/mario.gif)<br><br>

YOU CAN CREATE DYNAMIC ANIMATIONS OF YOUR MODELS

![VISUALIZATIONS](https://github.com/HCB06/PyerualJetwork/blob/main/Media/fit_history.gif)<br>
![VISUALIZATIONS](https://github.com/HCB06/PyerualJetwork/blob/main/Media/neuron_history.gif)<br>
![VISUALIZATIONS](https://github.com/HCB06/PyerualJetwork/blob/main/Media/neural_web.gif)<br>

YOU CAN CREATE AND VISUALIZE YOUR MODEL ARCHITECTURE

![VISUALIZATIONS](https://github.com/HCB06/PyerualJetwork/blob/main/Media/model_arc.png)<br>
![VISUALIZATIONS](https://github.com/HCB06/PyerualJetwork/blob/main/Media/eval_metrics.png)<br>



HOW DO I IMPORT IT TO MY PROJECT?

Anaconda users can access the 'Anaconda Prompt' terminal from the Start menu and add the necessary library modules to the Python module search queue by typing "pip install pyerualjetwork" and pressing enter. If you are not using Anaconda, you can simply open the 'cmd' Windows command terminal from the Start menu and type "pip install PyerualJetwork". (Visual Studio Code reccomended) After installation, it's important to periodically open the terminal of the environment you are using and stay up to date by using the command "pip install PyerualJetwork --upgrade".

After installing the module using "pip" you can now call the library module in your project environment. Use: “from pyerualjetwork.cpu import nn. Now, you can call the necessary functions from the nn module.

The PLAN algorithm & ENE algorithm will not be explained in this document. This document focuses on how professionals can integrate and use PyerualJetwork in their systems. However, briefly, the PLAN algorithm can be described as a classification algorithm. PLAN algorithm achieves this task with an incredibly energy-efficient, fast, and hyperparameter-free user-friendly approach. For more detailed information, you can check out ![PYERUALJETWORK USER MANUEL](https://github.com/HCB06/PyerualJetwork/blob/main/Welcome_to_PyerualJetwork/PYERUALJETWORK_USER_MANUEL_AND_LEGAL_INFORMATION(EN).pdf) file.