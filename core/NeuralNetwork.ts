// core/NeuralNetwork.ts
import { Layer } from './Layer';
import type { Optimizer } from './Optimizer';

export class NeuralNetwork {
  public layers: Layer[];
  private weightGrads: number[][][] = [];
  private biasGrads: number[][] = [];

  constructor(layerSizes: number[]) {
    this.layers = [];
    for (let i = 1; i < layerSizes.length; i++) {
      const isOutput = i === layerSizes.length - 1;
      this.layers.push(new Layer(layerSizes[i-1], layerSizes[i], isOutput));
    }
  }

  forward(input: number[]): number[] {
    let output = [...input];
    for (const layer of this.layers) {
      output = layer.forward(output);
      if (output.some(v => !Number.isFinite(v))) {
        throw new Error('Invalid value in forward pass');
      }
    }
    return output;
  }

  backward(gradient: number[]): void {
    this.weightGrads = [];
    this.biasGrads = [];
    
    let currentGrad = [...gradient];
    for (const layer of [...this.layers].reverse()) {
      currentGrad = layer.backward(currentGrad);
      
      if (layer.lastOutput) {
        const weightGrad = layer.weights.map((row, i) => 
          row.map((_, j) => currentGrad[i] * layer.lastOutput![j])
        );
        const biasGrad = currentGrad;
        
        this.weightGrads.unshift(weightGrad);
        this.biasGrads.unshift(biasGrad);
      }
    }
  }

  updateWeights(optimizer: Optimizer) {
    this.layers.forEach((layer, i) => {
      layer.updateWeights(optimizer, this.weightGrads[i], this.biasGrads[i]);
    });
  }
}
