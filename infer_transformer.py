import numpy as np
import tensorflow as tf
import logging
# Import the TransformerModel, TransformerEncoderLayer, and Vocabulary classes from train_transformer.py
from train_transformer import TransformerModel, TransformerEncoderLayer, Vocabulary

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Parameters
MODEL_PATH = "transformer_model.keras"  # Path to the saved model
MAX_SEQ_LEN = 256  # Maximum sequence length

# Load the trained model
def load_model():
    logging.info(f"Loading model from {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects={"TransformerModel": TransformerModel, "TransformerEncoderLayer": TransformerEncoderLayer})
    logging.info("Model loaded successfully.")
    return model

# Tokenize input text
def tokenize(text):
    return text.lower().split()

# Pad sequences
def pad_sequences(sequences, maxlen):
    return tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=maxlen, padding='post', truncating='post')

# Updated chat function with context window and batching for efficiency
def chat_with_model(model, vocab):
    logging.info("Chat session started. Type 'exit' to end the session.")
    context = []  # Maintain a context window
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            logging.info("Chat session ended.")
            break

        # Tokenize and encode user input
        tokens = tokenize(user_input)
        encoded = [vocab.word2idx.get(word, vocab.word2idx["<unk>"]) for word in tokens]
        context.extend(encoded)

        # Trim context to fit within the maximum sequence length
        if len(context) > MAX_SEQ_LEN:
            context = context[-MAX_SEQ_LEN:]

        # Pad the context
        padded_context = pad_sequences([context], maxlen=MAX_SEQ_LEN)

        # Generate predictions
        predictions = model.predict(padded_context, batch_size=1)  # Use batch_size for efficiency
        response = decode_response(predictions, vocab)

        # Add model response to context
        response_tokens = tokenize(response)
        response_encoded = [vocab.word2idx.get(word, vocab.word2idx["<unk>"]) for word in response_tokens]
        context.extend(response_encoded)

        print(f"Bot: {response}")

# Improved decoding logic with nucleus sampling (top-p sampling)
def decode_response(predictions, vocab, top_p=0.9, temperature=1.0):
    logits = predictions[0] / temperature  # Apply temperature scaling
    sorted_indices = np.argsort(logits, axis=-1)[:, ::-1]  # Sort tokens by probability in descending order
    cumulative_probs = np.cumsum(np.sort(logits, axis=-1)[:, ::-1], axis=-1)

    # Mask tokens with cumulative probability above top_p
    mask = cumulative_probs > top_p
    sorted_indices[mask] = -1

    response_tokens = []
    for row in sorted_indices:
        valid_indices = row[row != -1]  # Filter out masked tokens
        if len(valid_indices) == 0:
            continue
        sampled_idx = np.random.choice(valid_indices)  # Sample from valid tokens
        if sampled_idx == vocab.word2idx["<pad>"] or sampled_idx >= len(vocab.idx2word):
            continue
        response_tokens.append(vocab.idx2word[sampled_idx])

    return ' '.join(response_tokens)

# Main script
if __name__ == "__main__":
    # Load the model
    model = load_model()

    # Example vocabulary (replace with actual vocabulary used during training)
    vocab = Vocabulary()
    vocab.word2idx = {"<pad>": 0, "<unk>": 1, "hello": 2, "world": 3}  # Example mapping
    vocab.idx2word = ["<pad>", "<unk>", "hello", "world"]

    # Start chat session
    chat_with_model(model, vocab)