import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, GlobalAveragePooling1D, Dense, Input, Dropout
from tensorflow.keras.models import Model
import os

DATA_PATH = "tr_data/cabinet_dataset_google/gemini-2.5-flash-preview2.json"
MAX_VOCAB_SIZE = 10000
MAX_SEQUENCE_LENGTH = 200
BATCH_SIZE = 32
EPOCHS = 10
MODEL_PATH = "cabinet_model№3.h5"
TOKENIZER_PATH = "cabinet_tokenizer№3.json"

with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

valid_entries = [
    entry for entry in data
    if isinstance(entry.get("labels"), dict) and len(entry["labels"]) == 23 and isinstance(entry.get("article_text"), str)
]

texts = [entry["article_text"] for entry in valid_entries]
y = np.array([list(entry["labels"].values()) for entry in valid_entries], dtype=np.float32)

X_train_texts, X_test_texts, y_train, y_test = train_test_split(texts, y, test_size=0.2, random_state=42)

tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train_texts)

with open(TOKENIZER_PATH, "w", encoding="utf-8") as f:
    f.write(tokenizer.to_json())

X_train_seq = tokenizer.texts_to_sequences(X_train_texts)
X_test_seq = tokenizer.texts_to_sequences(X_test_texts)
X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')

inputs = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
x = Embedding(MAX_VOCAB_SIZE, 128)(inputs)
x = Bidirectional(LSTM(64, return_sequences=True))(x)
x = GlobalAveragePooling1D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)
outputs = Dense(23, activation="sigmoid")(x)

model = Model(inputs, outputs)
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),
    optimizer="adam",
    metrics=[tf.keras.metrics.AUC(name="AUC", multi_label=True), "accuracy"]
)

model.summary()

history = model.fit(
    X_train_pad,
    y_train,
    validation_data=(X_test_pad, y_test),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS
)


plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(history.history['AUC'], label='Train AUC')
plt.plot(history.history['val_AUC'], label='Val AUC')
plt.title('AUC')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.legend()

plt.tight_layout()
plt.savefig("training_plot.png")
plt.show()

model.save(MODEL_PATH)
print(f"\n✅ Model saved to: {MODEL_PATH}")
print(f"✅ Tokenizer saved to: {TOKENIZER_PATH}")
