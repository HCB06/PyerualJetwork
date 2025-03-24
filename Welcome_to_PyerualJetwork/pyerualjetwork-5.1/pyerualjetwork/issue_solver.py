""" 


Issue Solver
============
This module provides ready-to-use functions to identify potential issues caused by version incompatibilities in major updates,
ensuring users are not affected by such problems. PyereualJetwork aims to offer a seamless experience for its users.


Module functions:
-----------------
- update_model_to_v5()

Examples: https://github.com/HCB06/PyerualJetwork/tree/main/Welcome_to_PyerualJetwork/ExampleCodes

PyerualJetwork document: https://github.com/HCB06/PyerualJetwork/blob/main/Welcome_to_PyerualJetwork/PYERUALJETWORK_USER_MANUEL_AND_LEGAL_INFORMATION(EN).pdf

- Author: Hasan Can Beydili
- YouTube: https://www.youtube.com/@HasanCanBeydili
- Linkedin: https://www.linkedin.com/in/hasan-can-beydili-77a1b9270/
- Instagram: https://www.instagram.com/canbeydilj
- Contact: tchasancan@gmail.com
"""

def update_model_to_v5(model_name, model_path, is_cuda):

    """
    update_model_to_v5 function helps users for update models from older versions to newer versions.
   
    :param str model_name: Name of saved model.
    
    :param str model_path: Path of saved model.

    :param bool is_cuda: If model saved with cuda modules.
    
    :return: prints terminal if succes.
    """

    if is_cuda:

        from .model_operations_cuda import (get_act, 
                                            get_weights,
                                            get_scaler, 
                                            get_acc, 
                                            get_model_type,
                                            get_weights_type, 
                                            get_weights_format,
                                            load_model,
                                            save_model)
    else:

        from .model_operations_cpu import (get_act,
                                            get_weights,
                                            get_scaler, 
                                            get_acc, 
                                            get_model_type, 
                                            get_weights_type,
                                            get_weights_format,
                                            load_model,
                                            save_model)

    model = load_model(model_name, model_path)

    activations = model[get_act()]
    weights = model[get_weights()]
    scaler_params = model[get_scaler()]
    test_acc = model[get_acc()]
    model_type = model[get_model_type()]
    weights_type = model[get_weights_type()]
    weights_format = model[get_weights_format()]

    from .__init__ import __version__
    device_version = __version__

    save_model("updated_" + model_name, weights, model_type, scaler_params, test_acc, model_path, activations, weights_type, weights_format)

    print(f"\nModel succesfully updated to {device_version}. NOTE: This operation just for compatibility. You may still have perfomance issues in this situation please install model's version of pyerualjetwork.")