from pyerualjetwork.cpu.activation_functions import all_activations


def activation_potentiation():

    activations_list = all_activations()

    print('All available activations: ',  activations_list, "\n\nYOU CAN COMBINE EVERY ACTIVATION. EXAMPLE: ['linear', 'tanh'] or ['waveakt', 'linear', 'sine'].")

    return activations_list

def docs_and_examples():

    print('PLAN & ENE document: https://github.com/HCB06/PyerualJetwork/tree/main/Welcome_to_PLAN\n')
    print('PLAN examples: https://github.com/HCB06/PyerualJetwork/tree/main/Welcome_to_PyerualJetwork/ExampleCodes\n')
    print('ENE examples: https://github.com/HCB06/PyerualJetwork/tree/main/Welcome_to_PyerualJetwork/ExampleCodes/ENE\n')
    print('PyerualJetwork document and examples: https://github.com/HCB06/PyerualJetwork/tree/main/Welcome_to_PyerualJetwork')