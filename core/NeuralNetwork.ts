import { Layer } from './Layer';
import type { ActivationFunction } from './Activation';
import type { Optimizer } from './Optimizer';

export class NeuralNetwork {
  public layers: Layer[];

  constructor(layerSizes: number[], activation?: ActivationFunction) {
    this.layers = [];
    for (let i = 1; i < layerSizes.length; i++) {
      this.layers.push(new Layer(layerSizes[i-1], layerSizes[i], activation));
    }
  }

  forward(input: number[]): number[] {
    return this.layers.reduce((acc, layer) => layer.forward(acc), input);
  }

  backward(gradient: number[]): number[] {
    return this.layers.reduceRight((acc, layer) => layer.backward(acc), gradient);
  }

  updateWeights(optimizer: Optimizer) {
    this.layers.forEach(layer => layer.updateWeights(optimizer));
  }
}
