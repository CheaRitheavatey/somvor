import React, { useState } from "react";
import api from "./api";  

export default function App() {
  const [sequence, setSequence] = useState(""); // raw text input
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handlePredict = async () => {
    try {
      setLoading(true);
      // Parse the text area into a 2D array
      const parsed = JSON.parse(sequence);
      const res = await api.post("/predict", { sequence: parsed });
      setResult(res.data);
    } catch (err) {
      console.error(err);
      setResult({ error: "Invalid input or server error" });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={styles.container}>
      <h1>ASL Sign Prediction</h1>
      <p>Paste a landmark array (e.g. [[x1,y1,...],[x2,y2,...],...])</p>
      <textarea
        style={styles.textarea}
        rows={8}
        placeholder='Example: [[0.1,0.2,...],[0.3,0.4,...]]'
        value={sequence}
        onChange={(e) => setSequence(e.target.value)}
      />
      <button style={styles.button} onClick={handlePredict} disabled={loading}>
        {loading ? "Predicting..." : "Send to Model"}
      </button>

      {result && (
        <div style={styles.resultBox}>
          {result.error ? (
            <p style={{ color: "red" }}>{result.error}</p>
          ) : (
            <>
              <h3>Prediction: {result.prediction}</h3>
              <p>Confidence: {(result.confidence * 100).toFixed(2)}%</p>
            </>
          )}
        </div>
      )}
    </div>
  );
}

const styles = {
  container: {
    maxWidth: 600,
    margin: "40px auto",
    padding: 20,
    fontFamily: "sans-serif",
    textAlign: "center",
    border: "1px solid #ddd",
    borderRadius: 8,
  },
  textarea: {
    width: "100%",
    padding: 10,
    marginBottom: 20,
    fontFamily: "monospace",
  },
  button: {
    padding: "10px 20px",
    fontSize: 16,
    cursor: "pointer",
    backgroundColor: "#4CAF50",
    color: "#fff",
    border: "none",
    borderRadius: 4,
  },
  resultBox: {
    marginTop: 20,
    padding: 10,
    border: "1px solid #ccc",
    borderRadius: 4,
    background: "#f9f9f9",
  },
};
