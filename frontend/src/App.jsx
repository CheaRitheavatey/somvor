import React, { useRef, useState, useEffect } from 'react';
import * as handPoseDetection from '@tensorflow-models/hand-pose-detection';
import '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-webgl';

function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [detector, setDetector] = useState(null);
  const [isWebcamStarted, setIsWebcamStarted] = useState(false);
  const [gestureResult, setGestureResult] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [modelInfo, setModelInfo] = useState(null);
  const [sequence, setSequence] = useState([]);

  // Initialize hand detector and get model info
  useEffect(() => {
    const initialize = async () => {
      setIsLoading(true);
      
      try {
        // Get model info from backend
        const response = await fetch('http://localhost:8000/info');
        if (response.ok) {
          const info = await response.json();
          setModelInfo(info);
          console.log("Model info:", info);
        }
        
        // Initialize hand detector
        const model = handPoseDetection.SupportedModels.MediaPipeHands;
        const detectorConfig = {
          runtime: 'tfjs',
          modelType: 'full'
        };
        const detectorInstance = await handPoseDetection.createDetector(model, detectorConfig);
        setDetector(detectorInstance);
      } catch (error) {
        console.error('Initialization error:', error);
      }
      
      setIsLoading(false);
    };

    initialize();
  }, []);

  // Start webcam
  const startWebcam = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { width: 640, height: 480 } 
      });
      videoRef.current.srcObject = stream;
      setIsWebcamStarted(true);
      
      // Start detection loop once video is playing
      videoRef.current.onplaying = () => {
        detectGestures();
      };
    } catch (error) {
      console.error('Error accessing webcam:', error);
    }
  };

  // Process hand landmarks to create feature vector
  const processLandmarks = (keypoints) => {
    // Convert keypoints to a flat array of coordinates
    const features = [];
    
    for (const keypoint of keypoints) {
      features.push(keypoint.x);
      features.push(keypoint.y);
      features.push(keypoint.z || 0);
    }
    
    return features;
  };

  // Detect gestures and send to backend
  const detectGestures = async () => {
    if (!detector || !isWebcamStarted || !modelInfo) return;

    const context = canvasRef.current.getContext('2d');
    context.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);

    // Estimate hands
    const hands = await detector.estimateHands(videoRef.current);
    
    // Draw hand landmarks
    if (hands.length > 0) {
      drawHands(hands, context);
      
      // Process landmarks to create feature vector
      const features = processLandmarks(hands[0].keypoints);
      
      // Add to sequence
      setSequence(prev => {
        const newSequence = [...prev, features];
        
        // Keep only the required sequence length
        if (newSequence.length > modelInfo.seq_len) {
          return newSequence.slice(-modelInfo.seq_len);
        }
        return newSequence;
      });
    }

    // Continue detection
    requestAnimationFrame(detectGestures);
  };

  // Send sequence to backend for prediction
  useEffect(() => {
    const sendSequenceForPrediction = async () => {
      if (sequence.length === modelInfo.seq_len) {
        try {
          const response = await fetch('http://localhost:8000/predict_sequence', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({ sequence }),
          });
          
          if (response.ok) {
            const result = await response.json();
            setGestureResult(`${result.prediction} (${(result.confidence * 100).toFixed(1)}% confidence)`);
          }
        } catch (error) {
          console.error('Error sending data to backend:', error);
        }
      }
    };

    if (modelInfo && sequence.length === modelInfo.seq_len) {
      sendSequenceForPrediction();
    }
  }, [sequence, modelInfo]);

  // Draw hand landmarks on canvas
  const drawHands = (hands, context) => {
    context.strokeStyle = 'red';
    context.fillStyle = 'red';
    
    for (const hand of hands) {
      for (const keypoint of hand.keypoints) {
        context.beginPath();
        context.arc(keypoint.x, keypoint.y, 5, 0, 2 * Math.PI);
        context.fill();
      }
      
      // Draw connections between keypoints
      const connections = handPoseDetection.util.keypointsConnections;
      for (const connection of connections) {
        const [start, end] = connection;
        context.beginPath();
        context.moveTo(hand.keypoints[start].x, hand.keypoints[start].y);
        context.lineTo(hand.keypoints[end].x, hand.keypoints[end].y);
        context.stroke();
      }
    }
  };

  return (
    <div className="app">
      <h1>Hand Gesture Recognition</h1>
      
      <div className="controls">
        <button 
          onClick={startWebcam} 
          disabled={isWebcamStarted || isLoading || !modelInfo}
        >
          {isLoading ? 'Loading Model...' : 'Start Webcam'}
        </button>
      </div>
      
      <div className="camera-container">
        <video 
          ref={videoRef} 
          autoPlay 
          playsInline 
          width="640" 
          height="480" 
          style={{ display: isWebcamStarted ? 'block' : 'none' }}
        />
        <canvas 
          ref={canvasRef} 
          width="640" 
          height="480" 
          style={{ position: 'absolute', left: 0, top: 0 }}
        />
      </div>
      
      {modelInfo && (
        <div className="model-info">
          <p>Model expects: {modelInfo.seq_len} frames with {modelInfo.feature_dim} features each</p>
          <p>Sequence progress: {sequence.length}/{modelInfo.seq_len}</p>
        </div>
      )}
      
      {gestureResult && (
        <div className="result">
          <h2>Detected Gesture: {gestureResult}</h2>
        </div>
      )}
      
      {!isWebcamStarted && (
        <div className="instructions">
          <p>Click "Start Webcam" to begin gesture recognition</p>
          <p>Make sure your FastAPI backend is running on port 8000</p>
        </div>
      )}
    </div>
  );
}

export default App;