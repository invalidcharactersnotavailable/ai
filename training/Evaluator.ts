import { NeuralNetwork } from '../core/NeuralNetwork';
import type { DataPoint } from '../data/Dataset';
import { MSELoss } from '../core/Loss';

interface Metrics {
  accuracy: number;
  loss: number;
}

export class Evaluator {
  static evaluate(model: NeuralNetwork, dataset: DataPoint[]): Metrics {
    let correct = 0;
    let totalLoss = 0;
    const loss = new MSELoss();

    for (const { input, target } of dataset) {
      const output = model.forward(input);
      totalLoss += loss.calculate(output, target);
      
      if (Math.round(output[0]) === target[0]) {
        correct++;
      }
    }

    return {
      accuracy: correct / dataset.length,
      loss: totalLoss / dataset.length
    };
  }
}
