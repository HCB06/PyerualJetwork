from pyerualjetwork.activation_functions_cpu import all_activations


def activation_potentiation():

    activations_list = all_activations()

    print('All available activations: ',  activations_list, "\n\nYOU CAN COMBINE EVERY ACTIVATION. EXAMPLE: ['linear', 'tanh'] or ['waveakt', 'linear', 'sine'].")

    return activations_list

def docs_and_examples():

    print('PLAN document: https://github.com/HCB06/Anaplan/tree/main/Welcome_to_PLAN\n')
    print('PLAN examples: https://github.com/HCB06/Anaplan/tree/main/Welcome_to_PyerualJetwork/ExampleCodes\n')
    print('PLANEAT examples: https://github.com/HCB06/Anaplan/tree/main/Welcome_to_PyerualJetwork/ExampleCodes/ENE\n')
    print('Anaplan document and examples: https://github.com/HCB06/Anaplan/tree/main/Welcome_to_Anaplan')