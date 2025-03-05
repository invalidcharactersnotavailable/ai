// training/Trainer.ts
import { NeuralNetwork } from '../core/NeuralNetwork';
import { LossFunction } from '../core/Loss';
import { Dataset } from '../data/Dataset';
import { Adam } from '../core/Optimizer';
import { Hyperparams } from '../core/Hyperparams';

export class Trainer {
  constructor(
    private model: NeuralNetwork,
    private loss: LossFunction,
    private optimizer: Adam
  ) {}

  train(dataset: Dataset) {
    for (let epoch = 0; epoch < Hyperparams.epochs; epoch++) {
      const batches = dataset.createBatches(Hyperparams.batchSize);
      
      for (const batch of batches) {
        let totalLoss = 0;
        
        this.model.backward([]); // Initialize gradients

        for (const dataPoint of batch) {
          const output = this.model.forward(dataPoint.input);
          totalLoss += this.loss.calculate(output, dataPoint.target);
          const gradient = this.loss.gradient(output, dataPoint.target);
          this.model.backward(gradient);
        }

        this.model.updateWeights(this.optimizer);
        
        // Learning rate decay
        if (epoch % 1000 === 0) {
          this.optimizer.learningRate *= 0.9;
        }
      }
    }
  }
}
