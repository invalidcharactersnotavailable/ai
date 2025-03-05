// app.ts
import { NeuralNetwork } from './core/NeuralNetwork';
import { Trainer } from './training/Trainer';
import { Adam } from './core/Optimizer';
import { CrossEntropyLoss } from './core/Loss';
import { Dataset } from './data/Dataset';

// Initialize dataset
const trainingData = new Dataset([
  { input: [0, 0], target: [0] },
  { input: [0, 1], target: [1] },
  { input: [1, 0], target: [1] },
  { input: [1, 1], target: [0] },
]);
trainingData.normalize();

// Initialize model and trainer
const model = new NeuralNetwork([2, 4, 1]);
const optimizer = new Adam();
const trainer = new Trainer(model, new CrossEntropyLoss(), optimizer);

// Train
trainer.train(trainingData);

// Test
console.log('XOR Predictions:');
trainingData.data.forEach(dataPoint => {
  try {
    const prediction = model.forward(dataPoint.input)[0];
    console.log(
      `Input: [${dataPoint.input.map(x => x.toFixed(2))}] =>`,
      `${prediction.toFixed(4)} (${prediction > 0.5 ? '1' : '0'})`
    );
  } catch (error) {
    console.error('Prediction failed:', error);
  }
});
