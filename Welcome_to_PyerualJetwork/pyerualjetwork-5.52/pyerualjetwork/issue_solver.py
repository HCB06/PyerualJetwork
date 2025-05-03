""" 


Issue Solver
============
This module provides ready-to-use functions to identify potential issues caused by version incompatibilities in major updates,
ensuring users are not affected by such problems. PyereualJetwork aims to offer a seamless experience for its users.


Module functions:
-----------------
- update_model_to_v5_4()

Examples: https://github.com/HCB06/PyerualJetwork/tree/main/Welcome_to_PyerualJetwork/ExampleCodes

PyerualJetwork document: https://github.com/HCB06/PyerualJetwork/blob/main/Welcome_to_PyerualJetwork/PYERUALJETWORK_USER_MANUEL_AND_LEGAL_INFORMATION(EN).pdf

- Author: Hasan Can Beydili
- YouTube: https://www.youtube.com/@HasanCanBeydili
- Linkedin: https://www.linkedin.com/in/hasan-can-beydili-77a1b9270/
- Instagram: https://www.instagram.com/canbeydilj
- Contact: tchasancan@gmail.com
"""

def update_model_to_v5_4(model_name, model_path, is_cuda):

    """
    update_model_to_v5_4 function helps users for update models from older versions to newer versions.
   
    :param str model_name: Name of saved model.
    
    :param str model_path: Path of saved model.

    :param bool is_cuda: If model saved with cuda modules.
    
    :return: prints terminal if succes.
    """

    from .model_ops import save_model

    if is_cuda:

        from .old_cuda_model_ops import (get_act, 
                                            get_weights,
                                            get_scaler, 
                                            get_acc, 
                                            get_model_type,
                                            get_weights_type, 
                                            get_weights_format,
                                            get_model_df,
                                            get_act_pot,
                                            get_model_version,
                                            load_model)
    else:

        from .old_cpu_model_ops import (get_act,
                                            get_weights,
                                            get_scaler, 
                                            get_acc, 
                                            get_model_type, 
                                            get_weights_type,
                                            get_weights_format,
                                            get_model_df,
                                            get_act_pot,
                                            get_model_version,
                                            load_model)

    model = load_model(model_name, model_path)

    activations = model[get_act()]
    weights = model[get_weights()]
    scaler_params = model[get_scaler()]
    test_acc = model[get_acc()]
    model_type = model[get_model_type()]
    weights_type = model[get_weights_type()]
    weights_format = model[get_weights_format()]
    model_df = model[get_model_df()]
    act_pot = model[get_act_pot()]
    version = model[get_model_version()]

    from .model_ops import get_model_template

    template_model = get_model_template()

    model = template_model(weights,
                            None,
                            test_acc,
                            activations,
                            scaler_params,
                            None,
                            model_type,
                            weights_type,
                            weights_format,
                            device_version,
                            model_df,
                            act_pot
                            )
    

    from .__init__ import __version__
    device_version = __version__

    save_model(model, "updated_" + model_name, model_path)

    print(f"\nModel succesfully updated from {version} to {device_version}. In this location: {model_path}\nNOTE: This operation just for compatibility. You may still have perfomance issues in this situation please install model's version of pyerualjetwork.")