// src/components/SignPredictor.jsx
import React, { useEffect, useRef, useState } from "react";
import API from "../api";
import { Hands } from "@mediapipe/hands";
import { Camera } from "@mediapipe/camera_utils";

export default function SignPredictor() {
  const videoRef = useRef(null);
  const cameraRef = useRef(null);
  const bufferRef = useRef([]); // sliding buffer of frames
  const lastSentRef = useRef(0);
  const [seqLen, setSeqLen] = useState(null);
  const [featureDim, setFeatureDim] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [history, setHistory] = useState([]);
  const [running, setRunning] = useState(false);

  // fetch model info on mount
  useEffect(() => {
    API.get("/info").then(res => {
      setSeqLen(res.data.seq_len);
      setFeatureDim(res.data.feature_dim);
      console.log("[info]", res.data);
    }).catch(err => {
      console.error("Failed to fetch /info", err);
    });
  }, []);

  // helper to convert Mediapipe landmarks -> feature vector
  const landmarksToVector = (landmarks) => {
    // landmarks is array of 21 {x,y,z}
    if (!landmarks) {
      // return zeros
      return new Array(featureDim).fill(0.0);
    }
    const out = [];
    for (let i = 0; i < landmarks.length; i++) {
      const lm = landmarks[i];
      out.push(lm.x ?? 0);
      out.push(lm.y ?? 0);
      out.push(lm.z ?? 0);
    }
    // Trim or pad to featureDim
    if (out.length > featureDim) {
      return out.slice(0, featureDim);
    } else if (out.length < featureDim) {
      return out.concat(new Array(featureDim - out.length).fill(0.0));
    }
    return out;
  };

  useEffect(() => {
    if (!seqLen || !featureDim) return;

    // setup MediaPipe Hands
    const hands = new Hands({
      locateFile: (file) => `/hands/${file}`
    });

    hands.setOptions({
      maxNumHands: 1,
      selfieMode: true,
      minDetectionConfidence: 0.6,
      minTrackingConfidence: 0.5,
    });

    hands.onResults((results) => {
      // results.multiHandLandmarks is array of hand landmarks or undefined
      const landmarks = (results.multiHandLandmarks && results.multiHandLandmarks[0]) || null;
      const vec = landmarksToVector(landmarks); // length featureDim
      // push to buffer (sliding)
      bufferRef.current.push(vec);
      if (bufferRef.current.length > seqLen) bufferRef.current.shift();

      // if buffer full, send once every 600ms
      const now = Date.now();
      if (bufferRef.current.length === seqLen && (now - lastSentRef.current > 600)) {
        // send
        const payload = { sequence: bufferRef.current.slice() }; // copy
        lastSentRef.current = now;
        API.post("/predict_sequence", payload)
          .then(res => {
            const { prediction, confidence } = res.data;
            setPrediction({ label: prediction, confidence });
            setHistory(h => [{ label: prediction, confidence, ts: Date.now() }, ...h].slice(0, 50));
          })
          .catch(err => {
            console.error("predict error", err);
          });
      }
    });

    // set up camera using @mediapipe/camera_utils
    const videoElem = videoRef.current;
    cameraRef.current = new Camera(videoElem, {
      onFrame: async () => {
        await hands.send({ image: videoElem });
      },
      width: 640,
      height: 480
    });

    return () => {
      hands.close();
    };
  }, [seqLen, featureDim]);

  // start / stop handlers
  const handleStart = async () => {
    if (!videoRef.current) return;
    try {
      // ask for camera permissions and attach stream to video element
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      videoRef.current.srcObject = stream;
      await videoRef.current.play();
      // start camera utils
      cameraRef.current && cameraRef.current.start();
      setRunning(true);
    } catch (err) {
      console.error("camera start failed", err);
      alert("Camera start failed: " + err.message);
    }
  };
  const handleStop = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const tracks = videoRef.current.srcObject.getTracks();
      tracks.forEach(t => t.stop());
      videoRef.current.srcObject = null;
    }
    cameraRef.current && cameraRef.current.stop && cameraRef.current.stop();
    bufferRef.current = [];
    setRunning(false);
  };

  return (
    <div style={{display:'flex',gap:20}}>
      <div>
        <div style={{border:'1px solid #ddd', width:640, height:480, borderRadius:8, overflow:'hidden'}}>
          <video ref={videoRef} style={{width:'100%',height:'100%',objectFit:'cover'}} playsInline muted />
        </div>
        <div style={{marginTop:8, display:'flex', gap:8}}>
          <button onClick={handleStart} disabled={running || !seqLen}>Start</button>
          <button onClick={handleStop} disabled={!running}>Stop</button>
          <div style={{marginLeft:12}}>
            {seqLen ? `seq_len=${seqLen}, feat_dim=${featureDim}` : 'loading model info...'}
          </div>
        </div>
      </div>

      <div style={{width:320}}>
        <div style={{padding:12, border:'1px solid #eee', borderRadius:8}}>
          <h3>Prediction</h3>
          <div style={{minHeight:60}}>
            {prediction ? (
              <>
                <div style={{fontSize:18, fontWeight:700}}>{prediction.label}</div>
                <div style={{color:'#666'}}>{(prediction.confidence*100).toFixed(1)}%</div>
              </>
            ) : <div style={{color:'#999'}}>No prediction yet</div>}
          </div>
        </div>

        <div style={{marginTop:12, padding:12, border:'1px solid #eee', borderRadius:8}}>
          <h4>History</h4>
          <ul style={{maxHeight:300, overflowY:'auto', paddingLeft:14}}>
            {history.map((h, i) => (
              <li key={i}>
                {new Date(h.ts).toLocaleTimeString()} — <strong>{h.label}</strong> ({(h.confidence*100).toFixed(1)}%)
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}
