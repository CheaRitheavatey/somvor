const video = document.getElementById('video');
const predictBtn = document.getElementById('predictBtn');
const resultDiv = document.getElementById('result');

// get webcam stream once
async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
  } catch (err) {
    console.error('Camera error:', err);
  }
}
startCamera();

predictBtn.addEventListener('click', async () => {
  resultDiv.textContent = 'Predicting...';
  try {
    // call your FastAPI/Flask endpoint
    const res = await fetch('http://localhost:8000/predict', {
      method: 'POST'
    });
    const data = await res.json();
    resultDiv.textContent = 'Prediction: ' + (data.prediction || JSON.stringify(data));
  } catch (err) {
    console.error(err);
    resultDiv.textContent = 'Error contacting backend';
  }
});
