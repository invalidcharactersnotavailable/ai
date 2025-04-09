
# AI Command Guide

This guide provides the basic commands for training and interacting with the AI model using the `bun` tool.

## Commands

### 1. Train a Model
```bash
bun train --workers [workers]
```
Starts training the model using the specified number of workers. The `[workers]` argument determines the number of parallel workers for faster training.

### 2. Chat with the Model
```bash
bun chat
```
Runs the trained model in the terminal, allowing you to interact with it through chatting.

### 3. Chat with Reinforcement Training
```bash
bun chat --train
```
Runs the model in the terminal with chatting and reinforcement training. This allows the model to be trained on the fly by rewarding or punishing responses based on certain criteria.

### 4. Scrape Data from Project Gutenberg
```bash
bun scrape --count [count]
```
Scrapes data from Project Gutenberg. The `[count]` argument specifies the number of texts to scrape. Set a finite number to limit the scrape, for example, `--count 100` to scrape 100 texts.

### 5. Sanitize Text Data
```bash
bun sanitize --input [input_file] --output [output_file]
```
Sanitizes text data by removing unwanted characters, formatting, or non-text elements. Useful for cleaning up raw text data before use in training.

### 6. Convert Text to Training Data
```bash
bun convert --input [input_file] --output [output_file]
```
Converts raw text files into training data format. This allows you to use books or other text sources as input for model training.

## Notes
- Make sure to adjust the number of workers for optimal training based on your system capabilities.
- Reinforcement training can help improve the model over time, but it's important to define appropriate rewards and punishments for better results.
- The scraper can be used to gather books and texts for training or other purposes, with finite count control via `--count`.
- Use the sanitizer to clean your text data before converting it to training data format.
