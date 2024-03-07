// prediction.js
const fs = require('fs');
const tf = require('@tensorflow/tfjs-node');
const { promisify } = require('util');
const { loadLayersModel } = require('@tensorflow/tfjs');

const modelPath = 'path/to/model/model.h5';

// Convert Keras HDF5 model to TensorFlow.js format
async function convertModel() {
  const model = await loadLayersModel(`file://${modelPath}`);
  const savePath = 'path/to/model/model.json';
  await model.save(`file://${savePath}`);
  console.log(`Model converted and saved to ${savePath}`);
}

// Run the conversion once (comment out this line after running it once)
// convertModel();

// Load the converted model
const convertedModelPath = 'path/to/model/model.json';
const model = await tf.loadLayersModel(`file://${convertedModelPath}`);

const classNames = [
  'Alpukat', 'Apel', 'Bawang Bombai', 'Bawang Merah',
  'Bawang Putih', 'Bayam', 'Beras Merah', 'Beras Putih',
  'Brokoli', 'Buah Naga', 'Buncis', 'Daging Ayam',
  'Daging Sapi', 'Daun Bawang', 'Edamame', 'Ikan Salmon',
  'Ikan Tuna', 'Jagung', 'Kacang Hijau', 'Kacang Kedelai',
  'Kacang Merah', 'Kacang Polong', 'Kentang', 'Labu Putih',
  'Labu Siam', 'Melon', 'Oatmeal', 'Pakcoy', 'Pepaya', 'Pisang',
  'Tauge', 'Telur Ayam', 'Telur Puyuh', 'Tomat', 'Udang', 'Wortel'
];

async function predict(imageBuffer) {
  const imageArray = new Float32Array(imageBuffer);
  const imageTensor = tf.tensor4d(imageArray, [1, 150, 150, 3]);
  const predictions = model.predict(imageTensor);
  const predictedClass = predictions.argMax(1).dataSync()[0];

  const predictionLabel = classNames[predictedClass];
  const confidenceLevel = predictions.dataSync()[predictedClass];

  return { prediction: predictionLabel, confidence: confidenceLevel };
}

module.exports = { predict };
