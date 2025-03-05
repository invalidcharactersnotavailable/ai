import { NeuralNetwork } from '../core/NeuralNetwork';
import { LossFunction } from '../core/Loss';
import { Dataset } from '../data/Dataset';
import { Optimizer } from '../core/Optimizer';
import { Hyperparams } from '../config/Hyperparams';

export class Trainer {
  constructor(
    private model: NeuralNetwork,
    private loss: LossFunction,
    private optimizer: Optimizer
  ) {}

  train(dataset: Dataset, epochs: number, batchSize: number = Hyperparams.batchSize) {
    for (let epoch = 0; epoch < epochs; epoch++) {
      const batches = dataset.createBatches(batchSize);
      for (const batch of batches) {
        let totalLoss = 0;
        
        for (const dataPoint of batch) {
          const output = this.model.forward(dataPoint.input);
          totalLoss += this.loss.calculate(output, dataPoint.target);
          const gradient = this.loss.gradient(output, dataPoint.target);
          this.model.backward(gradient);
        }

        this.model.layers.forEach(layer => {
          layer.weights = layer.weights.map(row => 
            row.map(w => Math.max(
              -Hyperparams.gradientClipRange, 
              Math.min(Hyperparams.gradientClipRange, w)
            ))
          );
        });

        this.model.updateWeights(this.optimizer);
      }
    }
  }
}
