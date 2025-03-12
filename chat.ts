import { readFileSync } from "fs";
import * as readline from "readline";

interface markovModel {
  chain: { [key: string]: string[] };
  starters: string[];
}

const model: markovModel = JSON.parse(readFileSync("model.json", "utf-8"));

const randomChoice = <T>(arr: T[]): T => arr[Math.floor(Math.random() * arr.length)];

const generateResponse = (seed?: string, maxWords: number = 20): string => {
  let currentWord: string;
  if (seed && model.chain[seed]) {
    currentWord = seed;
  } else {
    currentWord = randomChoice(model.starters);
  }
  
  const responseWords: string[] = [currentWord];

  for (let i = 0; i < maxWords - 1; i++) {
    const nextWords = model.chain[currentWord];
    if (!nextWords || nextWords.length === 0) break;
    currentWord = randomChoice(nextWords);
    responseWords.push(currentWord);
    if (/[.!?]$/.test(currentWord)) break;
  }
  return responseWords.join(" ");
};

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

console.log("welcome to the ai chat. type your message (or 'exit' to quit) and press enter.");
rl.prompt();

rl.on("line", (line: string) => {
  const input = line.trim().toLowerCase();
  if (input === "exit") {
    console.log("goodbye, friend!");
    process.exit(0);
  }
  
  const inputWords = input.split(/\s+/);
  const seed = inputWords.find(word => model.chain[word]);
  
  const response = generateResponse(seed);
  console.log(response);
  rl.prompt();
});
