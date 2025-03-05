# ai

├── core/                        # Core AI functionality
│   ├── NeuralNetwork.ts         # Main NN class
│   ├── Layer.ts                 # Individual layer logic
│   ├── Activation.ts            # Activation functions (ReLU, Sigmoid, etc.)
│   ├── Loss.ts                  # Loss functions (MSE, Cross-Entropy, etc.)
│   ├── Optimizer.ts             # Optimizers (SGD, Adam, etc.)
│   ├── Index.ts                 # Export all core modules
├── data/                        # Dataset handling
│   ├── Dataset.ts               # Data loading and preprocessing
│   ├── Split.ts                 # Train-test split functions
├── training/                    # Training logic
│   ├── Trainer.ts               # Model training loop
│   ├── Evaluator.ts             # Performance metrics
├── utils/                       # Utility functions
│   ├── MathUtils.ts             # Matrix operations, randomization
│   ├── FileIO.ts                # Save/load model weights
├── config/                      # Configuration files
│   ├── Hyperparams.ts           # Learning rate, batch size, etc.
│── app.ts                       # Main entry point (initializes and runs AI)
│── tsconfig.json                # TypeScript config
