__version__ = "4.8"
__update__ = """* Changes: https://github.com/HCB06/PyerualJetwork/blob/main/CHANGES
* PyerualJetwork Homepage: https://github.com/HCB06/PyerualJetwork/tree/main
* PyerualJetwork document: https://github.com/HCB06/PyerualJetwork/blob/main/Welcome_to_PyerualJetwork/PYERUALJETWORK_USER_MANUEL_AND_LEGAL_INFORMATION(EN).pdf
* YouTube tutorials: https://www.youtube.com/@HasanCanBeydili"""

def print_version(version):
    print(f"PyerualJetwork Version {version}\n")

def print_update_notes(update):
    print(f"Notes:\n{update}")

print_version(__version__)
print_update_notes(__update__)

required_modules = ["scipy", "tqdm", "pandas", "numpy", "colorama", "cupy", "psutil"]

missing_modules = []
for module in required_modules:
    try:
        __import__(module)
    except ModuleNotFoundError:
        missing_modules.append(module)

if missing_modules:
    raise ImportError(
    f"Missing modules detected: {', '.join(missing_modules)}\n"
    "Please run the following command to install the missing packages:\n\n"
    f"    pip install {' '.join(missing_modules)}\n\n"
    "For more information, visit the PyerualJetwork GitHub README.md file:\n"
    "https://github.com/HCB06/PyerualJetwork/blob/main/README.md"

    )