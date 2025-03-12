import { readdirSync, readFileSync, writeFileSync, existsSync, mkdirSync } from 'fs';
import { join } from 'path';

interface MarkovModel {
  chain: { [key: string]: string[] };
  starters: string[];
  responses: { [key: string]: { [key: string]: number } };
}

const trainingDir = join(process.cwd(), 'training_data');
const cacheDir = 'model_caches';
const finalModelFile = 'final_model.json';

// Function to read .txt files from the training data directory
function getTxtFiles(dir: string): string[] {
  return readdirSync(dir).filter(file => file.endsWith('.txt'));
}

// Function to process a batch of files and return a Markov model
function processBatch(files: string[]): MarkovModel {
  const chain: { [key: string]: string[] } = {};
  const starters: string[] = [];
  const responses: { [key: string]: { [key: string]: number } } = {};

  files.forEach(file => {
    const filePath = join(trainingDir, file);
    const content = readFileSync(filePath, 'utf-8');
    const words = content.trim().split(/\s+/);

    let previousWord = '';
    words.forEach((word, index) => {
      if (index === 0 || /[.!?]$/.test(previousWord)) {
        starters.push(word);
      }
      if (!Array.isArray(chain[word])) {
        chain[word] = [];
      }
      if (index < words.length - 1) {
        const next = words[index + 1];
        chain[word].push(next);
      }
      // Capture response patterns dynamically
      if (index > 0 && previousWord.match(/[?]/)) {
        const question = words.slice(Math.max(0, index - 5), index).join(' ');
        responses[question] = responses[question] || {};
        responses[question][word] = (responses[question][word] || 0) + 1;
      }
      previousWord = word;
    });
  });

  return { chain, starters, responses };
}

// Function to merge multiple Markov models into one
function mergeModels(models: MarkovModel[]): MarkovModel {
  const mergedChain: { [key: string]: string[] } = {};
  const mergedStarters: string[] = [];
  const mergedResponses: { [key: string]: { [key: string]: number } } = {};

  models.forEach(model => {
    Object.entries(model.chain).forEach(([word, nextWords]) => {
      if (!mergedChain[word]) {
        mergedChain[word] = [];
      }
      mergedChain[word] = [...new Set([...mergedChain[word], ...nextWords])];
    });

    mergedStarters.push(...model.starters);
    Object.entries(model.responses).forEach(([question, responseMap]) => {
      if (!mergedResponses[question]) {
        mergedResponses[question] = {};
      }
      Object.entries(responseMap).forEach(([response, count]) => {
        mergedResponses[question][response] = (mergedResponses[question][response] || 0) + count;
      });
    });
  });

  return { chain: mergedChain, starters: mergedStarters, responses: mergedResponses };
}

// Main function to orchestrate the training process
async function main() {
  const files = getTxtFiles(trainingDir);
  console.log(`Found ${files.length} .txt files in the training data directory.`);

  // Ensure the cache directory exists
  if (!existsSync(cacheDir)) {
    mkdirSync(cacheDir);
  }

  const batchSize = 100;
  const totalBatches = Math.ceil(files.length / batchSize);
  const models: MarkovModel[] = [];

  for (let i = 0; i < totalBatches; i++) {
    const batchFiles = files.slice(i * batchSize, (i + 1) * batchSize);
    console.log(`Processing batch ${i + 1} of ${totalBatches}...`);

    const model = processBatch(batchFiles);
    models.push(model);

    // Save the model cache for this batch
    const cacheFile = join(cacheDir, `model_cache_batch_${i + 1}.json`);
    writeFileSync(cacheFile, JSON.stringify(model, null, 2));
    console.log(`Batch ${i + 1} processed and saved to ${cacheFile}.`);
  }

  // Merge all batch models into one
  console.log('Merging batch models into a single model...');
  const finalModel = mergeModels(models);

  // Save the final compiled model
  writeFileSync(finalModelFile, JSON.stringify(finalModel, null, 2));
  console.log(`Final model saved to ${finalModelFile}.`);
}

main().catch(err => console.error(err));
