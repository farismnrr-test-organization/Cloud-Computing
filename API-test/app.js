// main.js
const express = require('express');
const multer = require('multer');
const sharp = require('sharp');
const fs = require('fs');

const { predict } = require('./prediction');

const app = express();
const port = 3000;

const storage = multer.memoryStorage();
const upload = multer({ storage: storage });

app.get('/', (req, res) => {
  res.json({ "Hello": "World" });
});

app.post('/predict', upload.single('file'), async (req, res) => {
  try {
    const buffer = req.file.buffer;
    const image = await sharp(buffer).resize(150, 150).toBuffer();
    const { prediction, confidence } = await predict(image);

    res.json({ prediction, confidence: confidence.toString() });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Internal Server Error' });
  }
});

app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});
