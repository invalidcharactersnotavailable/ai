// core/Activation.ts
import { Hyperparams } from './Hyperparams'; // Add this import

export interface ActivationFunction {
  forward(inputs: number[]): number[];
  backward(gradients: number[], outputs: number[]): number[];
}

export class LeakyReLU implements ActivationFunction {
  forward(inputs: number[]): number[] {
    return inputs.map(x => Math.max(x * Hyperparams.leakyReluAlpha, x));
  }

  backward(gradients: number[], outputs: number[]): number[] {
    return outputs.map((o, i) => 
      o > 0 ? gradients[i] : Hyperparams.leakyReluAlpha * gradients[i]
    );
  }
}

export class Sigmoid implements ActivationFunction {
  forward(inputs: number[]): number[] {
    return inputs.map(x => 1 / (1 + Math.exp(-x)));
  }

  backward(gradients: number[], outputs: number[]): number[] {
    return outputs.map((o, i) => {
      const grad = gradients[i];
      return grad * o * (1 - o) + Hyperparams.epsilon;
    });
  }
}
