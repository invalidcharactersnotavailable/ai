export interface ActivationFunction {
  forward(inputs: number[]): number[];
  backward(gradients: number[]): number[];
}

export class ReLU implements ActivationFunction {
  forward(inputs: number[]): number[] {
    return inputs.map(x => Math.max(0, x));
  }

  backward(gradients: number[]): number[] {
    return gradients.map(g => g > 0 ? 1 : 0);
  }
}

export class Sigmoid implements ActivationFunction {
  forward(inputs: number[]): number[] {
    return inputs.map(x => 1 / (1 + Math.exp(-x)));
  }

  backward(gradients: number[]): number[] {
    const outputs = this.forward(gradients);
    return outputs.map(g => g * (1 - g));
  }
}
