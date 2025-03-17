import tensorflow as tf
import numpy as np

# Data
corpus = ["I like playing football with my friends",
          "I enjoy playing tennis",
          "I hate swimming",
          "I love basketball"]

# Hyperparameters
window_size = 3
embedding_dim = 50
batch_size = 16
epochs = 100
learning_rate = 0.01

# Tokenize the corpus
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(corpus)
vocab_size = len(tokenizer.word_index) + 1

# Generate training data
sequences = tokenizer.texts_to_sequences(corpus)
data = []
for seq in sequences:
    for i in range(len(seq)):
        for j in range(max(0, i - window_size), min(len(seq), i + window_size + 1)):
            if i != j:
                data.append([seq[i], seq[j]])

# Convert data to numpy arrays
data = np.array(data)
x_train = data[:, 0]
y_train = data[:, 1]

# Model
inputs = tf.keras.layers.Input(shape=(1,))
embeddings = tf.keras.layers.Embedding(vocab_size, embedding_dim)(inputs)
flatten = tf.keras.layers.Flatten()(embeddings)
output = tf.keras.layers.Dense(vocab_size, activation='softmax')(flatten)

model = tf.keras.models.Model(inputs=inputs, outputs=output)

# Compile the model
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))

# Train the model
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

# Get the word embeddings
word_embeddings = model.get_layer(index=1).get_weights()[0]

# Function to get word vector
def get_vector(word):
    idx = tokenizer.word_index[word]
    return word_embeddings[idx]

# Example usage
word = "football"
vector = get_vector(word)
print(f"Vector representation of '{word}': {vector}")

#Context word
def get_context_words(word):
    idx = tokenizer.word_index[word]

    # Find the context indices within the specified window
    context_indices = list(range(max(0, idx - window_size), min(vocab_size, idx + window_size + 1)))
    context_words = [word for word, index in tokenizer.word_index.items() if index in context_indices]
    return context_words
# Example usage
focus_word = "playing"
context_words = get_context_words(focus_word)
print(f"Context words for '{focus_word}': {context_words}")
