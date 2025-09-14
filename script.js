let model;
const modelURL = "model/model.json"; // Path to your model
const imageUpload = document.getElementById("imageUpload");
const previewImg = document.getElementById("previewImg");
const predictBtn = document.getElementById("predictBtn");
const resultBox = document.getElementById("resultBox");
const startCamera = document.getElementById("startCamera");
const camera = document.getElementById("camera");

async function loadModel() {
  resultBox.textContent = "Loading model...";
  model = await tf.loadLayersModel(modelURL);
  resultBox.textContent = "✅ Model loaded. Upload or start camera!";
}
loadModel();

// Image upload
imageUpload.addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    previewImg.src = reader.result;
    previewImg.hidden = false;
    camera.hidden = true;
  };
  reader.readAsDataURL(file);
});

// Start camera
startCamera.addEventListener("click", async () => {
  const stream = await navigator.mediaDevices.getUserMedia({ video: true });
  camera.srcObject = stream;
  camera.hidden = false;
  previewImg.hidden = true;
});

// Predict
predictBtn.addEventListener("click", async () => {
  if (!model) return;

  let input;
  if (!previewImg.hidden) {
    input = tf.browser.fromPixels(previewImg).resizeBilinear([224, 224]).toFloat().div(255).expandDims();
  } else if (!camera.hidden) {
    input = tf.browser.fromPixels(camera).resizeBilinear([224, 224]).toFloat().div(255).expandDims();
  } else {
    resultBox.textContent = "Please upload an image or start the camera.";
    return;
  }

  const predictions = await model.predict(input).data();
  input.dispose();

  // Get metadata for labels (optional)
  const metadata = await fetch("model/metadata.json").then(r => r.json());
  const classes = metadata.labels || predictions.map((_, i) => `Class ${i}`);

  // Find best prediction
  const maxIndex = predictions.indexOf(Math.max(...predictions));
  const label = classes[maxIndex];
  const confidence = (predictions[maxIndex] * 100).toFixed(2);

  resultBox.textContent = `🎯 Prediction: ${label} (${confidence}%)`;
});
