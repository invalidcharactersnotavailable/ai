// utils/FileIO.ts
import { NeuralNetwork } from '../core/NeuralNetwork';
import { Layer } from '../core/Layer';
import { LeakyReLU, Sigmoid } from '../core/Activation'; // Changed from ReLU to LeakyReLU

export class FileIO {
  static async saveModel(model: NeuralNetwork, path: string): Promise<void> {
    const modelData = {
      layers: model.layers.map(layer => ({
        weights: layer.weights,
        biases: layer.biases,
        activation: layer.activation.constructor.name
      }))
    };
    await Bun.write(path, JSON.stringify(modelData));
  }

  static async loadModel(path: string): Promise<NeuralNetwork> {
    const data = await Bun.file(path).json();
    const model = new NeuralNetwork([]);
    
    model.layers = data.layers.map((layer: any) => {
      const newLayer = new Layer(0, 0);
      newLayer.weights = layer.weights;
      newLayer.biases = layer.biases;
      newLayer.activation = layer.activation === 'LeakyReLU' ? new LeakyReLU() : new Sigmoid();
      return newLayer;
    });
    
    return model;
  }
}
