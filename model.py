import pickle

model_path = '/Users/benothmane/Desktop/AIR_OPTIMISATION/upload/folder/Train_model2.pkl'

with open(model_path, 'rb') as f:
    model = pickle.load(f)

print(type(model))
