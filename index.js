const tf = require('@tensorflow/tfjs-node-gpu');
const fs = require('fs');

// Test TensorFlow.js
console.log('TFJS version:', tf.version.tfjs);

const a = tf.tensor([1, 2, 3]);
const b = tf.tensor([3, 2, 1]);
a.add(b).print(); // Should print [4, 4, 4]

// Define some input parameters
const vocabSize = 1000; // Number of words in our "dictionary"
const embeddingDim = 64; // Size of each word vector
const seqLength = 1; // How many words per input sequence

// Create and embedding layer. Maps each tokenID to a trainable vector
function EmbeddingLayer(vocabSize, embeddingDim) {
    return tf.layers.embedding({
        inputDim: vocabSize, 
        outputDim: embeddingDim,
        inputLength: seqLength
    });
}

// Add a simple self-attention mechanism
// We will use TensorflowJS Dense layers to approximate queries, keys, 
// and values used in attention
function SelfAttention(embeddingDim) {
    // Input for self attention layer
    const input = tf.input({
        shape: [seqLength, embeddingDim]
    });

    // Dense layers to create Query, Key, Value
    const query = tf.layers.dense({
        units: embeddingDim
    }).apply(input);
    const key = tf.layers.dense({
        units: embeddingDim
    }).apply(input);
    const value = tf.layers.dense({
        units: embeddingDim
    }).apply(input);

    // Scaled dot-product attention (simplified)
    const score = tf.layers.dot({ axes: -1 }).apply([query, key]); // score = dot([query, key])
    const weights = tf.layers.activation({ activation: 'softmax' }).apply(score); // weights = activation(score)

    const context = tf.layers.dot({ axes: [2, 1] }).apply([weights, value]); // context = dot([weights, value])

    return tf.model({
        inputs: input,
        outputs: context
    });
}
// This is a very simplified version just to illustrate the mechanics. A real GPT uses multi-head attention with more steps.
// Key Concepts:
//  Queries (Q): Represent the current token's "question".
//  Keys (K): Represent what each token "offers".
//  Values (V): Actual content of each token.
//  Output: Weighted sum of values, where weights are determined by query-key compatibility.

// Build the tiny GPT model
// function buildMiniGPT() {
//     // Main model input. Not sub layer
//     const input = tf.input({ shape: [seqLength] });

//     const x = EmbeddingLayer(vocabSize, embeddingDim).apply(input);
//     const attn = SelfAttention(embeddingDim).apply(x);

//     // Feed-forward layer
//     const dense = tf.layers.dense({
//         units: embeddingDim, 
//         activation: 'relu'
//     }).apply(attn);
//     const output = tf.layers.dense({
//         units: vocabSize, activation: 'softmax'
//     }).apply(dense);

//     return tf.model({ inputs: input, outputs: output });
// }

// const model = buildMiniGPT();
// model.summary();
// Build the 1-to-1 model (a simple bigram model)
function buildModel() {
    // Main model input. Expects shape [1]
    const input = tf.input({ shape: [seqLength] });

    // x shape will be [batch_size, 1, embeddingDim]
    const x = EmbeddingLayer(vocabSize, embeddingDim).apply(input);

    // Flatten to remove the sequence dimension
    // Shape becomes [batch_size, embeddingDim]
    const flattened = tf.layers.flatten().apply(x);

    // Feed-forward layer
    const dense = tf.layers.dense({
        units: embeddingDim, 
        activation: 'relu'
    }).apply(flattened);

    const output = tf.layers.dense({
        units: vocabSize, 
        activation: 'softmax'
    }).apply(dense);

    return tf.model({ inputs: input, outputs: output });
}

// const model = buildMiniGPT(); // BEFORE
const model = buildModel(); // AFTER
model.summary();

// Training time!
// Tokenize the training text
const text = fs.readFileSync('data/train.txt', 'utf8').split('\n');
const tokens = [...new Set(text.join(' ').split(' '))];

const wordIndex = {};
tokens.forEach((word, i) => wordIndex[word] = i + 1);  // Key = Word, value = number/index

console.log(wordIndex);
// { hello: 1, world: 2, there: 3, javascript: 4 }

// Create Input/Output Pairs for training for next word prediction
// const sequences = [];
// text.forEach(line => {
//     const words = line.split(' ');
//     for (let i = 0; i < words.length -1; i++) {
//         const input = wordIndex[words[i]];
//         const output = wordIndex[words[i+1]];
//         sequences.push({ input, output });
//     }
// });

// console.log('Sequences: ', sequences);
// // [ { input: 1, output: 2 }, { input: 1, output: 3 }, ... ]

// // Convert into tensors
// const xs = tf.tensor2d(sequences.map(s => [s.input]));
// const ys = tf.tensor2d(sequences.map(s => [s.output]));

// console.log('Tensor xs: ', xs);
// console.log('Tensor ys: ', ys);

// Combine all text into one long list of tokens
const allTokens = [];
text.forEach(line => {
    const words = line.split(' ');
    words.forEach(word => {
        if (word && wordIndex[word]) { 
            allTokens.push(wordIndex[word]);
        }
    });
});

// Create Input/Output Pairs
const sequences = [];
for (let i = 0; i < allTokens.length - 1; i++) {
    // Input is the current word, output is the next word
    sequences.push({ 
        input: allTokens[i], 
        output: allTokens[i + 1] 
    });
}

console.log('Sequences: ', sequences.slice(0, 3));
// [ { input: 1, output: 2 }, { input: 2, output: 3 }, ... ]

// Convert into tensors
// xs is shape [num_sequences, 1]
const xs = tf.tensor2d(sequences.map(s => [s.input]), [sequences.length, 1], 'float32');

// ys is shape [num_sequences] (a 1D list of correct labels)
// This is required for 'sparseCategoricalCrossentropy'
const ys = tf.tensor1d(sequences.map(s => s.output), 'float32'); // <-- FIX

console.log('Tensor xs shape:', xs.shape); // Should be [*, 1]
console.log('Tensor ys shape:', ys.shape); // Should be [*]


// Compile and train the model
model.compile({
    optimizer: 'adam',
    loss: 'sparseCategoricalCrossentropy',
    metrics: ['accuracy']
}); 

const fitModel = async () => {
    await model.fit(xs, ys, {
        epochs: 50,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                console.log(`Epoch ${epoch + 1}: loss=${logs.loss.toFixed(4)}`);
            }
        }
    });
};

// Running Inference
// let inputWord = "hello";
// let inputIndex = [wordIndex[inputWord]];

async function generateNextWord(inputIndex) {
    console.log('Predicting next word...');
    const inputTensor = tf.tensor2d([inputIndex], [1,1]);
    const prediction = model.predict(inputTensor);
    const probs = prediction.arraySync()[0];

    // Pick the word with the highest probability
    const predictedIndex = probs.indexOf(Math.max(...probs));
    return Object.keys(wordIndex).find(key => wordIndex[key] === predictedIndex);
}

// Build a short sequence by looping this process
async function generateText(startWord, length = 5) {
    console.log('Generating text...');
    let result = [startWord];
    let currentWord = startWord;

    for (let i = 0; i < length; i++) {
        const nextWord = await generateNextWord([wordIndex[currentWord]]);
        result.push(nextWord);
        currentWord = nextWord;
    }

    return result.join(' ');
}


fitModel().then(async () => {
    await model.save('file://./models/my-first-transformer').then(() => {
        console.log('Model Saved!');
    }).catch(err => console.log(err));
    generateText('explore', 6).then(res => console.log('Generated text: ', res)).catch(err => console.log(err));
}).catch(err => console.log(err));