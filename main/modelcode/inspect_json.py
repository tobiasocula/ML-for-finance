import json
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
models_path = os.path.join(os.path.dirname(script_dir), 'models')

with open(os.path.join(models_path, "hist.json"), "r") as f:
    d = json.load(f)

for modelname in d.keys():
    print()
    print('MODEL:', modelname)
    print()
    print('PARAMETERS:')
    for paramname, paramval in d[modelname]["params"].items():
        print(paramname, ':', paramval)
    print()
    print('VALIDATION ERRORS:')
    for err in d[modelname]["validation_loss_per_epoch"]:
        print(err)
    print()
    print('TRAINING ERRORS:')
    print(d[modelname]["training_loss_per_epoch"])


