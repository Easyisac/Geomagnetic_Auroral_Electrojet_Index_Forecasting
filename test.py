from modelBuilder import *
from model import *
from dataPreparation import *

X, Y = create_value_dataset(loockback=10)
builder = singleInputTransformer(X, Y)
model, history = train_model(builder, X, Y, './results/single', 'single')