import { NeuralNetwork } from './neuralNetwork';
import * as fs from 'fs';
import * as path from 'path';

function wordToInput(word: string): number[] {
    const alphabet = 'abcdefghijklmnopqrstuvwxyz';
    return word.toLowerCase().split('').map(char => alphabet.indexOf(char) / 26);
}

function outputToWord(output: number[]): string {
    const alphabet = 'abcdefghijklmnopqrstuvwxyz';
    return output.map(val => alphabet[Math.floor(val * 26)]).join('');
}

function loadTrainingData(folderPath: string): { input: string, output: string }[] {
    const trainingData: { input: string, output: string }[] = [];
    const files = fs.readdirSync(folderPath);

    for (const file of files) {
        if (path.extname(file) === '.txt') {
            const content = fs.readFileSync(path.join(folderPath, file), 'utf-8');
            const lines = content.split('\n');
            for (let i = 0; i < lines.length - 1; i += 2) {
                trainingData.push({
                    input: lines[i].trim(),
                    output: lines[i + 1].trim()
                });
            }
        }
    }

    return trainingData;
}

const nn = new NeuralNetwork(128, 512, 128);
const trainingData = loadTrainingData('./trainingData');

for (let i = 0; i < 10000; i++) {
    const data = trainingData[Math.floor(Math.random() * trainingData.length)];
    nn.train(wordToInput(data.input), wordToInput(data.output));
}

console.log(outputToWord(nn.feedforward(wordToInput('hello'))));
