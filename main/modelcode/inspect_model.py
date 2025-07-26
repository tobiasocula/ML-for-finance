import json
import os

model_name = "mult_candle_model_0"

script_dir = os.path.dirname(os.path.abspath(__file__))
models_path = os.path.join(os.path.dirname(script_dir), 'models')

with open(os.path.join(models_path, "hist.json"), "r") as f:
    d = json.load(f)

model_json = d[f"{model_name}"]
print('PARAMS')
for pkey, pval in model_json['params'].items():
    print(pkey, ':', pval)

print()
print('DESCRIPTION')
print(model_json['description'])