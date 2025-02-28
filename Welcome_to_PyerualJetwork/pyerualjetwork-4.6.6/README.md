# PyerualJetwork [![Socket Badge](https://socket.dev/api/badge/pypi/package/pyerualjetwork/4.0.6?artifact_id=tar-gz)](https://socket.dev/pypi/package/pyerualjetwork/overview/4.0.6/tar-gz) [![CodeFactor](https://www.codefactor.io/repository/github/hcb06/pyerualjetwork/badge)](https://www.codefactor.io/repository/github/hcb06/pyerualjetwork) [![PyPI Downloads](https://static.pepy.tech/badge/pyerualjetwork)](https://pepy.tech/projects/pyerualjetwork) + [![PyPI Downloads](https://static.pepy.tech/badge/anaplan)](https://pepy.tech/projects/anaplan)


[![PyPI Downloads](https://static.pepy.tech/badge/pyerualjetwork/month)](https://pepy.tech/projects/pyerualjetwork) [![PyPI Downloads](https://static.pepy.tech/badge/pyerualjetwork/week)](https://pepy.tech/projects/pyerualjetwork) [![PyPI version](https://img.shields.io/pypi/v/pyerualjetwork.svg)](https://pypi.org/project/pyerualjetwork/)

Note: anaplan old name of pyerualjetwork

![PyerualJetwork](https://github.com/HCB06/PyerualJetwork/blob/main/Media/pyerualjetwork_with_name.png)<br><br><br>

Libraries.io Page: https://libraries.io/pypi/pyerualjetwork

PyPi Page: https://pypi.org/project/pyerualjetwork/

GitHub Page: https://github.com/HCB06/PyerualJetwork

YouTube Tutorials: https://www.youtube.com/watch?v=6wMQstZ00is&list=PLNgNWpM7HbsBpCx2VTJ4SK9wcPyse-EHw

	pip install pyerualjetwork
	
	from pyerualjetwork import plan
	from pyerualjetwork import planeat
	from pyerualjetwork import data_operations
	from pyerualjetwork import model_operations
	
	from pyerualjetwork import plan_cuda
	from pyerualjetwork import planeat_cuda
	from pyerualjetwork import data_operations_cuda
	from pyerualjetwork import model_operations_cuda
	
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

PyerualJetwork is a machine learning library written in Python for professionals, incorporating advanced, unique, new, and modern techniques with optimized GPU acceleration. Its most important component is the PLAN (Potentiation Learning Artificial Neural Network) https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4862342. (THIS ARTICLE IS FIRST VERSION OF PLAN.) MODERN VERSION OF PLAN: https://github.com/HCB06/PyerualJetwork/blob/main/Welcome_to_PLAN/PLAN.pdf
Both the PLAN algorithm and the PyerualJetwork library were created by Author, and all rights are reserved by Author.
PyerualJetwork is free to use for commercial business and individual users.
As of 12/21/2024, the library includes PLAN and PLANEAT module, but other machine learning modules are expected to be added in the future.

PyerualJetwork ready for both eager execution(like PyTorch) and static graph(like Tensorflow) concepts because PyerualJetwork using only functions.
For example:

fit function only fits given training data(suitable for dynamic graph) but learner function learns and optimize entire architecture(suitable for static graph). Or more deeper eager executions PyerualJetwork have: feed_forward function, list of activation functions, loss functions. You can create your unique model architecture. Move your data to GPU or CPU or manage how much should in GPU, Its all up to you.
<br><br>

PyerualJetworket includes Plan Vision, NLPlan, PLANEAT and at the between of both, Deep Plan.<br>

PLAN VISION:<br>

![PLAN VISION](https://github.com/HCB06/PyerualJetwork/blob/main/Media/PlanVision.jpg)

You can create artificial intelligence models that perform computer vision tasks using the plan module:<br>

![AUTONOMOUS](https://github.com/HCB06/PyerualJetwork/blob/main/Media/autonomous.gif)<br><br><br>
![XRAY](https://github.com/HCB06/PyerualJetwork/blob/main/Media/chest_xray.png)<br><br><br>
![GENDER](https://github.com/HCB06/PyerualJetwork/blob/main/Media/gender_classification.png)<br><br><br>

NLPlan:<br>

![NLPLAN](https://github.com/HCB06/PyerualJetwork/blob/main/Media/NLPlan.jpg)<br>

You can create artificial intelligence models that perform natural language processing tasks using the plan module:

![PLAN VISION](https://github.com/HCB06/PyerualJetwork/blob/main/Media/NLP.gif)

PLANEAT:<br>

You can create artificial intelligence models that perform reinforcement learning tasks and genetic optimization tasks using the planeat module:

![PLANEAT](https://github.com/HCB06/PyerualJetwork/blob/main/Media/PLANEAT_1.gif)<br>
![PLANEAT](https://github.com/HCB06/PyerualJetwork/blob/main/Media/PLANEAT_2.gif)<br>
![PLANEAT](https://github.com/HCB06/PyerualJetwork/blob/main/Media/mario.gif)<br><br>

YOU CAN CREATE DYNAMIC ANIMATIONS OF YOUR MODELS

![VISUALIZATIONS](https://github.com/HCB06/PyerualJetwork/blob/main/Media/fit_history.gif)<br>
![VISUALIZATIONS](https://github.com/HCB06/PyerualJetwork/blob/main/Media/neuron_history.gif)<br>
![VISUALIZATIONS](https://github.com/HCB06/PyerualJetwork/blob/main/Media/neural_web.gif)<br>

YOU CAN CREATE AND VISUALIZE YOUR MODEL ARCHITECTURE

![VISUALIZATIONS](https://github.com/HCB06/PyerualJetwork/blob/main/Media/model_arc.png)<br>
![VISUALIZATIONS](https://github.com/HCB06/PyerualJetwork/blob/main/Media/eval_metrics.png)<br>



HOW DO I IMPORT IT TO MY PROJECT?

Anaconda users can access the 'Anaconda Prompt' terminal from the Start menu and add the necessary library modules to the Python module search queue by typing "pip install pyerualjetwork" and pressing enter. If you are not using Anaconda, you can simply open the 'cmd' Windows command terminal from the Start menu and type "pip install PyerualJetwork". (Visual Studio Code reccomended) After installation, it's important to periodically open the terminal of the environment you are using and stay up to date by using the command "pip install PyerualJetwork --upgrade".

After installing the module using "pip" you can now call the library module in your project environment. Use: “from pyerualjetwork import plan”. Now, you can call the necessary functions from the plan module.

The PLAN algorithm will not be explained in this document. This document focuses on how professionals can integrate and use PyerualJetwork in their systems. However, briefly, the PLAN algorithm can be described as a classification algorithm. PLAN algorithm achieves this task with an incredibly energy-efficient, fast, and hyperparameter-free user-friendly approach. For more detailed information, you can check out ![PYERUALJETWORK USER MANUEL](https://github.com/HCB06/PyerualJetwork/blob/main/Welcome_to_PyerualJetwork/PYERUALJETWORK_USER_MANUEL_AND_LEGAL_INFORMATION(EN).pdf) file.
