import { Hyperparams } from './config/Hyperparams';
import { NeuralNetwork } from './core/NeuralNetwork';
import { Trainer } from './training/Trainer';
import { Adam } from './core/Optimizer';
import { MSELoss } from './core/Loss';
import { Dataset } from './data/Dataset';
import { ReLU } from './core/Activation';
import { Evaluator } from './training/Evaluator';

// Initialize dataset
const trainingData = new Dataset([
  { input: [0, 0], target: [0] },
  { input: [0, 1], target: [1] },
  { input: [1, 0], target: [1] },
  { input: [1, 1], target: [0] },
]);
trainingData.normalize();

// Initialize model with hyperparameters
const model = new NeuralNetwork(
  [Hyperparams.inputSize, Hyperparams.hiddenSize, Hyperparams.outputSize],
  new ReLU()
);

const optimizer = new Adam();
const trainer = new Trainer(model, new MSELoss(), optimizer);

// Train the model
trainer.train(trainingData, Hyperparams.epochs);

// Evaluate results
const metrics = Evaluator.evaluate(model, trainingData.data);
console.log(`Training Accuracy: ${(metrics.accuracy * 100).toFixed(1)}%`);
console.log(`Final Loss: ${metrics.loss.toFixed(4)}`);

// Test predictions
trainingData.data.forEach((dataPoint, index) => {
  const prediction = model.forward(dataPoint.input)[0];
  console.log(
    `Input: [${dataPoint.input.map(x => x.toFixed(1))}] => ${prediction.toFixed(4)}`
  );
});
