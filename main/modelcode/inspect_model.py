import keras

model_name = 'DaySin_OCHL_model.keras'

model = keras.models.load_model('../models/' + model_name)
model.summary()
print()
print('LAYERS')
for layer in model.layers:
    print('STATS')
    print(layer.name, layer.count_params())
    print()
    print('CONFIG')
    print(layer.get_config())
    print()

# get weights: model.get_weights()