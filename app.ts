import { NeuralNetwork } from './core/NeuralNetwork';
import { Trainer } from './training/Trainer';
import { Adam } from './core/Optimizer';
import { MSELoss } from './core/Loss';
import { Dataset } from './data/Dataset';
import { ReLU } from './core/Activation';

// Create and prepare dataset
const trainingData = new Dataset([
  { input: [0, 0], target: [0] },
  { input: [0, 1], target: [1] },
  { input: [1, 0], target: [1] },
  { input: [1, 1], target: [0] },
]);
trainingData.normalize();

// Initialize model with proper configuration
const model = new NeuralNetwork([2, 4, 1], new ReLU());
const optimizer = new Adam(0.001);
const trainer = new Trainer(model, new MSELoss(), optimizer);

// Train the model
trainer.train(trainingData, 2000, 2);

// Test predictions
console.log('XOR Predictions:');
trainingData.data.forEach((dataPoint, idx) => {
  const prediction = model.forward(dataPoint.input)[0];
  console.log(`Input: [${dataPoint.input}] => ${prediction.toFixed(4)}`);
});
