from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Setting Up
setup(
    name="pyerualjetwork",
    version="4.8",
    author="Hasan Can Beydili",
    author_email="tchasancan@gmail.com",
    description=(
        "PyerualJetwork is a machine learning library supported with GPU(CUDA) acceleration written in Python for professionals and researchers including with PLAN algorithm, PLANEAT algorithm (genetic optimization). Also includes data pre-process and memory manegament\n"
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

