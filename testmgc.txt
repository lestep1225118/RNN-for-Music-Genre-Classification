import os
import mido
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Define the path to the parent folder containing subfolders for each genre
parent_folder = "C:\\Users\\leand\\Downloads\\MGC\\adl-shortened"
'''
# Read the MIDI files from each genre subfolder and parse them into a NumPy array
dataset = []
labels = []
for genre_folder in os.listdir(parent_folder):
    if not os.path.isdir(os.path.join(parent_folder, genre_folder)):
        continue
    genre_label = 0 if genre_folder == "Electronic" else 1
    for filename in os.listdir(os.path.join(parent_folder, genre_folder)):
        if filename.endswith(".mid"):
            midi_file = mido.MidiFile(os.path.join(parent_folder, genre_folder, filename))
            notes = []
            for msg in midi_file:
                if msg.type == "note_on":
                    notes.append(msg.note)
            dataset.append(notes)
            labels.append(genre_label)
dataset = np.array(dataset, dtype=object)
labels = np.array(labels)

new_dataset = [song[:100] for song in dataset if len(song) > 100]
new_labels = [label for label, song in zip(labels, dataset) if len(song) > 100]
new_dataset = np.array(new_dataset)
new_labels = np.array(new_labels)

np.savetxt("dataset.csv", new_dataset, delimiter=",")
np.savetxt("labels.csv", new_labels, delimiter=",")
'''
new_dataset = np.loadtxt("C:\\Users\\leand\\Downloads\\MGC\\dataset.csv", delimiter=",")
new_labels = np.loadtxt("C:\\Users\\leand\\Downloads\\MGC\\labels.csv", delimiter=",")

new_dataset = np.reshape(new_dataset, (len(new_dataset), 100, 1))

# Create a TensorFlow model
model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(50,activation="relu"),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

# Compile model
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train model using the fit() method
if os.path.isfile("model.h5"):
    # Load saved model if it exists
    model = tf.keras.models.load_model("model.h5")
else:
    # Train a new model if saved model does not exist
    model.fit(new_dataset, new_labels, epochs=200, batch_size=None)
    model.save("model.h5")

# Define the path to the MIDI file you want to make a prediction for
midi_file_path = "C:\\Users\\leand\\Downloads\\MGC\\2_of_a_kind_jp.mid"

midi_file = mido.MidiFile(midi_file_path)
notes = []
for msg in midi_file:
    if msg.type == "note_on":
        notes.append(msg.note)

max_seq_length = 100
if len(notes) > max_seq_length:
    notes = notes[:max_seq_length]

input_seq = np.asarray(notes, dtype=np.int32)
input_seq = np.reshape(input_seq, (1, len(input_seq), 1))
input_seq = tf.convert_to_tensor(input_seq, dtype=tf.float32)

prediction = model.predict(input_seq)
new_label = prediction.round()[0][0]
if prediction < 0.5:
    print("electronic")
else:
    print("jazz")

# saliency map
def get_saliency_map(model, input_seq):
    with tf.GradientTape() as tape:
        tape.watch(input_seq)
        output = model(input_seq)
    gradients = tape.gradient(output, input_seq)
    saliency_map = np.abs(gradients.numpy())[0]
    return saliency_map

saliency_map = get_saliency_map(model, input_seq)
print(saliency_map)

# optimization
input_seq = tf.squeeze(input_seq)
delta_var = tf.Variable(tf.zeros(input_seq.shape))
other_label = 1-new_label
predict = lambda x: tf.squeeze(model(tf.expand_dims(x,0)))
loss = lambda: (predict(input_seq+delta_var)-other_label)**2
opt = tf.keras.optimizers.Adam(learning_rate=0.1)
for _ in range(11):
    tmp = opt.minimize(loss,var_list=[delta_var])
    print(predict(input_seq+delta_var).numpy())

delta = delta_var.numpy()

pred = predict(input_seq)
pred_label = "electronic" if pred < 0.5 else "jazz"
print(" prediction before: %.2f (%s)" % (pred,pred_label))

pred = predict(input_seq+delta)
pred_label = "electronic" if pred < 0.5 else "jazz"
print(" prediction after: %.2f (%s)" % (pred,pred_label))

