__version__ = "4.3.6"
__update__ = "* Changes: https://github.com/HCB06/PyerualJetwork/blob/main/CHANGES\n* PyerualJetwork Homepage: https://github.com/HCB06/PyerualJetwork/tree/main\n* PyerualJetwork document: https://github.com/HCB06/PyerualJetwork/blob/main/Welcome_to_PyerualJetwork/PYERUALJETWORK_USER_MANUEL_AND_LEGAL_INFORMATION(EN).pdf\n* YouTube tutorials: https://www.youtube.com/@HasanCanBeydili"

def print_version(__version__):
    print(f"PyerualJetwork Version {__version__}" + '\n')

def print_update_notes(__update__):
    print(f"Notes:\n{__update__}")

print_version(__version__)
print_update_notes(__update__)