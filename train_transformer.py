import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Dense, LayerNormalization, MultiHeadAttention, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.saving import register_keras_serializable
import logging
import pickle

# Simplified parameter configuration
DATA_FOLDER = "dataset/data"  # Folder containing .txt files
VOCAB_SIZE = 8192  # Most frequent tokens
MAX_SEQ_LEN = 1024  # Context window (tokens)
NUM_LAYERS = 4  # Transformer blocks
EMBEDDING_DIM = 256  # Token embedding size
NUM_HEADS = 2  # Attention heads per layer
FEEDFORWARD_DIM = EMBEDDING_DIM * 4  # Inner FF dimension
BATCH_SIZE = 4  # Sequences per device
EPOCHS = 15  # Passes over dataset
LEARNING_RATE = 1e-3  # Initial learning rate
MODEL_SAVE_PATH = "model.keras"  # Save in Keras format

# Ensure the optimizer is available
try:
    from tensorflow.keras.optimizers import AdamW
except ImportError:
    from tensorflow_addons.optimizers import AdamW

# Configure logging more granularly
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Simple word-level tokenizer
def tokenize(text):
    return text.lower().split()

# Build vocabulary
class Vocabulary:
    def __init__(self):
        self.word2idx = {"<pad>": 0, "<unk>": 1}
        self.idx2word = ["<pad>", "<unk>"]
        self.word_freq = {}

    def add_sentence(self, sentence):
        for word in sentence:
            self.word_freq[word] = self.word_freq.get(word, 0) + 1

    def build_vocab(self):
        sorted_words = sorted(self.word_freq.items(), key=lambda x: x[1], reverse=True)
        for word, _ in sorted_words[:VOCAB_SIZE - len(self.idx2word)]:
            if word not in self.word2idx:
                self.word2idx[word] = len(self.idx2word)
                self.idx2word.append(word)

    def encode(self, sentence):
        return [self.word2idx.get(word, self.word2idx["<unk>"]) for word in sentence]

    def __len__(self):
        return len(self.idx2word)

# Load and preprocess data
def load_data(folder, vocab):
    samples = []
    for filename in os.listdir(folder):
        try:
            with open(os.path.join(folder, filename), 'r', encoding='utf-8') as f:
                tokens = tokenize(f.read())
                samples.append(vocab.encode(tokens))
        except UnicodeDecodeError:
            try:
                with open(os.path.join(folder, filename), 'r', encoding='latin-1') as f:
                    tokens = tokenize(f.read())
                    samples.append(vocab.encode(tokens))
            except Exception as e:
                logging.warning(f"Skipping file {filename} due to encoding issues: {e}")
    return samples

# Pad sequences
def pad_sequences(sequences, maxlen):
    return tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=maxlen, padding='post', truncating='post')

# Transformer Encoder Layer
@register_keras_serializable(package="Custom", name="TransformerEncoderLayer")
class TransformerEncoderLayer(Model):
    def __init__(self, embedding_dim, num_heads, hidden_dim, dropout_rate=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim)
        self.ffn = tf.keras.Sequential([
            Dense(hidden_dim, activation='relu'),
            Dense(embedding_dim)
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

    def call(self, x, training):
        attn_output = self.mha(x, x, x)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        return {
            "embedding_dim": self.embedding_dim,
            "num_heads": self.num_heads,
            "hidden_dim": self.hidden_dim,
            "dropout_rate": self.dropout_rate
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# Transformer Model
@register_keras_serializable(package="Custom", name="TransformerModel")
class TransformerModel(Model):
    def __init__(self, vocab_size, embedding_dim, num_heads, hidden_dim, num_layers, max_seq_len):
        super(TransformerModel, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.positional_encoding = get_positional_encoding(max_seq_len, embedding_dim)
        self.enc_layers = [TransformerEncoderLayer(embedding_dim, num_heads, hidden_dim) for _ in range(num_layers)]
        self.dropout = Dropout(0.1)
        self.fc = Dense(vocab_size)

    def call(self, x, training=None):
        seq_len = tf.shape(x)[1]
        assert_op = tf.debugging.assert_less_equal(seq_len, self.max_seq_len, message=f"Sequence length {seq_len} exceeds maximum allowed {self.max_seq_len}")
        with tf.control_dependencies([assert_op]):
            # Ensure all components are cast to the same data type
            x = tf.cast(x, self.embedding.compute_dtype)
            positional_encoding = tf.cast(self.positional_encoding[:seq_len, :], self.embedding.compute_dtype)
            embedding_output = self.embedding(x)
            x = embedding_output + positional_encoding

        logging.info(f"Sequence length: {seq_len}")
        logging.info(f"Input shape after embedding: {x.shape}")

        x = self.dropout(x, training=training)
        for enc_layer in self.enc_layers:
            x = enc_layer(x, training=training)
        return self.fc(x)

    def get_config(self):
        return {
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
            "num_heads": self.num_heads,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "max_seq_len": self.max_seq_len
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# Optimize positional encoding for large MAX_SEQ_LEN
def get_positional_encoding(max_seq_len, embedding_dim):
    def positional_encoding_on_the_fly(position, i):
        angle_rate = 1 / np.power(10000, (2 * (i // 2)) / np.float32(embedding_dim))
        return np.sin(position * angle_rate) if i % 2 == 0 else np.cos(position * angle_rate)

    return tf.constant([[positional_encoding_on_the_fly(pos, i) for i in range(embedding_dim)] for pos in range(max_seq_len)], dtype=tf.float32)

# Adjust mixed precision policy based on hardware
policy = 'mixed_float16' if tf.config.list_physical_devices('GPU') else 'mixed_bfloat16'
tf.keras.mixed_precision.set_global_policy(policy)

# Add functionality to save the model after each epoch and resume training
checkpoint_dir = "checkpoints/"
os.makedirs(checkpoint_dir, exist_ok=True)

# Modify train_model_with_fit to save checkpoints after each epoch
def train_model_with_fit(model, dataset, vocab, epochs, validation_split=0.1, resume_from_checkpoint=None):
    # Ensure dataset is appropriately formatted for model.fit
    if not isinstance(dataset, tf.data.Dataset):
        dataset = tf.data.Dataset.from_tensor_slices(padded_samples).shuffle(len(padded_samples)).batch(BATCH_SIZE)

    # Prepare inputs and targets from the dataset
    inputs = []
    targets = []
    for batch in dataset:
        inputs.append(batch[:, :-1])
        targets.append(batch[:, 1:])

    inputs = tf.concat(inputs, axis=0)
    targets = tf.concat(targets, axis=0)

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    # Resume from checkpoint if specified
    initial_epoch = 0
    if resume_from_checkpoint:
        model = tf.keras.models.load_model(resume_from_checkpoint)
        logging.info(f"Resumed training from checkpoint: {resume_from_checkpoint}")
        initial_epoch = int(resume_from_checkpoint.split('-')[-1].split('.')[0])

    # Configure callbacks for checkpoints and logging
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, "model-epoch-{epoch:02d}.keras"),
        save_weights_only=False,
        save_freq='epoch'
    )
    logging_callback = tf.keras.callbacks.LambdaCallback(
        on_epoch_end=lambda epoch, logs: logging.info(f"Epoch {epoch + 1}/{epochs} - Loss: {logs['loss']:.4f} - Accuracy: {logs['accuracy']:.4f}")
    )

    # Split dataset into training and validation
    num_validation_samples = int(len(inputs) * validation_split)
    train_inputs, val_inputs = inputs[:-num_validation_samples], inputs[-num_validation_samples:]
    train_targets, val_targets = targets[:-num_validation_samples], targets[-num_validation_samples:]

    # Train the model
    model.fit(
        x=train_inputs,
        y=train_targets,
        validation_data=(val_inputs, val_targets),
        epochs=epochs,
        batch_size=BATCH_SIZE,
        callbacks=[checkpoint_callback, logging_callback],
        initial_epoch=initial_epoch
    )

    # Save the final model with optimizer state
    model.save(MODEL_SAVE_PATH, include_optimizer=True)
    logging.info(f"Model saved to {MODEL_SAVE_PATH}")

# Add functionality to fine-tune a trained model
def finetune_model(model_path, dataset, vocab, epochs):
    model = tf.keras.models.load_model(model_path)
    logging.info(f"Loaded model from {model_path} for fine-tuning.")
    train_model_with_fit(model, dataset, vocab, epochs)

# Main script
if __name__ == "__main__":
    # Build vocabulary
    vocab = Vocabulary()
    for filename in os.listdir(DATA_FOLDER):
        try:
            with open(os.path.join(DATA_FOLDER, filename), 'r', encoding='utf-8') as f:
                vocab.add_sentence(tokenize(f.read()))
        except UnicodeDecodeError:
            try:
                with open(os.path.join(DATA_FOLDER, filename), 'r', encoding='latin-1') as f:
                    vocab.add_sentence(tokenize(f.read()))
            except Exception as e:
                logging.warning(f"Skipping file {filename} due to encoding issues: {e}")
    vocab.build_vocab()  # Limit vocabulary size

    # Save the tokenizer
    TOKENIZER_PATH = "tokenizer.pkl"
    with open(TOKENIZER_PATH, "wb") as f:
        pickle.dump(vocab, f)
    logging.info(f"Tokenizer saved to {TOKENIZER_PATH}")

    # Load and preprocess data
    samples = load_data(DATA_FOLDER, vocab)
    padded_samples = pad_sequences(samples, maxlen=MAX_SEQ_LEN)

    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices(padded_samples)
    dataset = dataset.shuffle(len(padded_samples)).batch(BATCH_SIZE)

    # Initialize model
    model = TransformerModel(len(vocab), EMBEDDING_DIM, NUM_HEADS, FEEDFORWARD_DIM, NUM_LAYERS, MAX_SEQ_LEN)

    # Train the model
    train_model_with_fit(model, np.array(padded_samples), vocab, EPOCHS)