import json, os
from pathlib import Path

model_name = "multiple_stocks_model_5min_7"

root = Path.cwd()
models_path = root / 'main' / 'models'

with open(os.path.join(models_path, "model_info.json"), "r") as f:
    d = json.load(f)

model_json = d[f"{model_name}"]
print('PARAMS')
for pkey, pval in model_json['params'].items():
    print(pkey, ':', pval)

print()
print('DESCRIPTION')
print(model_json['description'])