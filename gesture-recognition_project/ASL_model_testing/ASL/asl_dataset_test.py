# load json and create model
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator

json_file = open('./my_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# loaded_model = load_model(loaded_model_json)
# load weights into new model
loaded_model.load_weights('./model.h5', 'r')
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])
data_dir = "./asl-alphabet/asl_alphabet_test"

target_size = (64, 64)
target_dims = (64, 64, 3) # add channel for RGB
n_classes = 29
val_frac = 0.0
batch_size = 64

data_augmentor = ImageDataGenerator(samplewise_center=True,
                                    samplewise_std_normalization=True,
                                    validation_split=val_frac)

train_generator = data_augmentor.flow_from_directory(data_dir, target_size=target_size)

scores = loaded_model.evaluate(train_generator)
print("Accuracy = ", scores[1])
