from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Setting Up
setup(
    name="pyerualjetwork",
    version="5.48",
    author="Hasan Can Beydili",
    author_email="tchasancan@gmail.com",
    description=(
        "PyereualJetwork is a GPU-accelerated machine learning library in Python for professionals and researchers. It features PLAN, MLP, Deep Learning training, and ENE (Eugenic NeuroEvolution) for genetic optimization, applicable to genetic algorithms or Reinforcement Learning (RL). The library includes data pre-processing, visualizations, model saving/loading, prediction, evaluation, training, and detailed or simplified memory management.\n"
        "* Changes: https://github.com/HCB06/PyerualJetwork/blob/main/CHANGES\n"
        "* PyerualJetwork document: "
        "https://github.com/HCB06/PyerualJetwork/blob/main/Welcome_to_PyerualJetwork/"
        "PYERUALJETWORK_USER_MANUEL_AND_LEGAL_INFORMATION(EN).pdf."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    keywords=[
        "model evaluation",
        "classification",
        "potentiation learning artificial neural networks",
        "NEAT",
        "genetic algorithms",
        "reinforcement learning",
        "neural networks",
    ],
)

