
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
bun scrape --count 0
```
Scrapes data from Project Gutenberg. The `--count 0` argument specifies scraping an unlimited number of texts (or a default number if interpreted that way).

## Notes
- Make sure to adjust the number of workers for optimal training based on your system capabilities.
- Reinforcement training can help improve the model over time, but it's important to define appropriate rewards and punishments for better results.
- The scraper can be used to gather books and texts for training or other purposes.
