import { mkdir, writeFile } from 'fs/promises';
import { join } from 'path';

const startId = 61;
const endId = 61000;
const baseUrl = 'https://www.gutenberg.org/cache/epub/';
const downloadDir = join(process.cwd(), 'training_data');

// ensure the download directory exists
async function ensureDownloadDir(): Promise<void> {
  await mkdir(downloadDir, { recursive: true });
}

// determine folder based on id pattern
function getFolder(id: number): string {
  return id < 10000 ? id.toString() : id.toString() + '0';
}

// construct download url for a given id
function constructUrl(id: number): string {
  const folder = getFolder(id);
  return `${baseUrl}${folder}/pg${id}.txt`;
}

// download a single file and save it
async function downloadFile(id: number): Promise<void> {
  const url = constructUrl(id);
  try {
    const res = await fetch(url);
    if (!res.ok) {
      console.error(`failed to download id ${id} from ${url}: ${res.statusText}`);
      return;
    }
    const arrayBuffer = await res.arrayBuffer();
    const fileName = `pg${id}.txt`;
    const filePath = join(downloadDir, fileName);
    await writeFile(filePath, Buffer.from(arrayBuffer));
    console.log(`downloaded: ${fileName}`);
  } catch (error) {
    console.error(`error downloading id ${id} from ${url}:`, error);
  }
}

// worker pool: continuously process ids from the shared array
async function downloadWorker(ids: number[], concurrency: number): Promise<void> {
  let index = 0;
  
  // each worker grabs the next id as soon as it finishes the current download
  const worker = async () => {
    while (true) {
      let currentIndex: number;
      // synchronize access to the shared index
      if (index < ids.length) {
        currentIndex = index;
        index++;
      } else {
        break;
      }
      await downloadFile(ids[currentIndex]);
    }
  };

  const workers: Promise<void>[] = [];
  for (let i = 0; i < concurrency; i++) {
    workers.push(worker());
  }
  await Promise.all(workers);
}

async function main() {
  await ensureDownloadDir();
  console.log(`starting optimized download from id ${startId} to ${endId}`);
  const ids: number[] = [];
  for (let i = startId; i <= endId; i++) {
    ids.push(i);
  }
  
  const concurrencyLimit = 50; // adjust this number as needed
  await downloadWorker(ids, concurrencyLimit);
  console.log('all downloads complete');
}

main().catch(console.error);
