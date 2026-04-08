import React, { useState, useRef, useEffect } from "react";
import { extractText, summarizeBert, summarizeKrylov, compareSummaries } from "./api/api";
import SummaryCard from "./components/SummaryCard";

function App() {
  // Shared states
  const [file, setFile] = useState(null);

  // Model A: BERT States
  const [bertResult, setBertResult] = useState(null);
  const [bertLoading, setBertLoading] = useState(false);
  const [bertElapsed, setBertElapsed] = useState(0);
  const bertIntervalRef = useRef(null);

  // Model B: Krylov States
  const [krylovResult, setKrylovResult] = useState(null);
  const [krylovLoading, setKrylovLoading] = useState(false);
  const [krylovElapsed, setKrylovElapsed] = useState(0);
  const krylovIntervalRef = useRef(null);

  // Similarity State
  const [similarity, setSimilarity] = useState(null);
  const [compLoading, setCompLoading] = useState(false);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  // Calculate similarity when both results are ready
  useEffect(() => {
    if (bertResult && krylovResult) {
      const getSimilarity = async () => {
        setCompLoading(true);
        try {
          const data = await compareSummaries(krylovResult.summary, bertResult.summary);
          setSimilarity(data);
        } catch (err) {
          console.error("Comparison failed", err);
        } finally {
          setCompLoading(false);
        }
      };
      getSimilarity();
    } else {
      setSimilarity(null);
    }
  }, [bertResult, krylovResult]);

  // BERT Execution Logic
  const runBertBench = async () => {
    if (!file) { alert("Select PDF first"); return; }

    setBertLoading(true);
    setBertResult(null);
    setBertElapsed(0);

    const startTime = Date.now();
    bertIntervalRef.current = setInterval(() => {
      setBertElapsed((Date.now() - startTime) / 1000);
    }, 10);

    try {
      const text = await extractText(file);
      const data = await summarizeBert(text);
      setBertResult(data);
    } catch (err) {
      console.error(err);
      alert("BERT Bench failed");
    } finally {
      clearInterval(bertIntervalRef.current);
      setBertLoading(false);
    }
  };

  // Krylov Execution Logic
  const runKrylovBench = async () => {
    if (!file) { alert("Select PDF first"); return; }

    setKrylovLoading(true);
    setKrylovResult(null);
    setKrylovElapsed(0);

    const startTime = Date.now();
    krylovIntervalRef.current = setInterval(() => {
      setKrylovElapsed((Date.now() - startTime) / 1000);
    }, 10);

    try {
      const text = await extractText(file);
      const data = await summarizeKrylov(text);
      setKrylovResult(data);
    } catch (err) {
      console.error(err);
      alert("Krylov Bench failed");
    } finally {
      clearInterval(krylovIntervalRef.current);
      setKrylovLoading(false);
    }
  };

  return (
    <div style={{
      backgroundColor: "#0f172a",
      minHeight: "100vh",
      padding: "40px",
      color: "white",
      fontFamily: "Inter, system-ui, Avenir, Helvetica, Arial, sans-serif"
    }}>
      <h1 style={{ textAlign: "center", marginBottom: "40px" }}>
        <span style={{ color: "#22d3ee" }}>Spectral</span> vs <span style={{ color: "#818cf8" }}>Baseline</span> Benchmark
      </h1>

      <div style={{
        display: "grid",
        gridTemplateColumns: "1fr 1fr",
        gap: "30px",
        maxWidth: "1200px",
        margin: "0 auto"
      }}>

        {/* BERT COLUMN */}
        <div style={{ backgroundColor: "#1e293b", padding: "20px", borderRadius: "16px" }}>
          <h2 style={{ color: "#94a3b8", marginBottom: "15px" }}>1. Baseline BERT</h2>
          <input type="file" accept=".pdf" onChange={handleFileChange} style={{ marginBottom: "15px", display: "block" }} />
          <button
            onClick={runBertBench}
            disabled={bertLoading}
            style={{
              width: "100%", padding: "12px", borderRadius: "8px", border: "none",
              backgroundColor: "#334155", color: "white", fontWeight: "bold", cursor: "pointer"
            }}
          >
            {bertLoading ? "Processing..." : "Run BERT Benchmark"}
          </button>

          <div style={{ marginTop: "20px", fontSize: "2rem", textAlign: "center", color: "#64748b" }}>
            {bertElapsed.toFixed(2)}s
          </div>

          {bertResult && (
            <div style={{ marginTop: "20px" }}>
              <SummaryCard title="BERT Output" summary={bertResult.summary} time={bertResult.time} />
            </div>
          )}
        </div>

        {/* KRYLOV COLUMN */}
        <div style={{ backgroundColor: "#1e293b", padding: "20px", borderRadius: "16px" }}>
          <h2 style={{ color: "#22d3ee", marginBottom: "15px" }}>2. Spectral Krylov-BERT</h2>
          <input type="file" accept=".pdf" onChange={handleFileChange} style={{ marginBottom: "15px", display: "block" }} />
          <button
            onClick={runKrylovBench}
            disabled={krylovLoading}
            style={{
              width: "100%", padding: "12px", borderRadius: "8px", border: "none",
              backgroundColor: "#22d3ee", color: "#0f172a", fontWeight: "bold", cursor: "pointer"
            }}
          >
            {krylovLoading ? "Processing..." : "Run Krylov Benchmark"}
          </button>

          <div style={{ marginTop: "20px", fontSize: "2rem", textAlign: "center", color: "#22d3ee" }}>
            {krylovElapsed.toFixed(2)}s
          </div>

          {krylovResult && (
            <div style={{ marginTop: "20px" }}>
              <SummaryCard title="Krylov-BERT Output" summary={krylovResult.summary} time={krylovResult.time} highlight />
            </div>
          )}
        </div>

      </div>

      {/* Comparison Metrics Section */}
      {(bertResult && krylovResult) && (
        <div style={{
          marginTop: "50px", maxWidth: "1200px", margin: "50px auto 0 auto", padding: "30px",
          backgroundColor: "rgba(34, 211, 238, 0.05)", borderRadius: "16px", border: "1px solid rgba(34, 211, 238, 0.2)"
        }}>
          <h2 style={{ color: "#22d3ee", textAlign: "center", marginBottom: "25px" }}>Analysis Results</h2>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "20px" }}>
            <div style={{ textAlign: "center", padding: "20px", background: "rgba(255,255,255,0.05)", borderRadius: "12px" }}>
              <h3 style={{ fontSize: "0.9rem", textTransform: "uppercase", color: "#94a3b8" }}>Performance Speedup</h3>
              <div style={{ fontSize: "2.5rem", fontWeight: "bold", color: "#22d3ee", marginTop: "10px" }}>
                {(bertResult.time / krylovResult.time).toFixed(1)}x Faster
              </div>
            </div>

            <div style={{ textAlign: "center", padding: "20px", background: "rgba(255,255,255,0.05)", borderRadius: "12px" }}>
              <h3 style={{ fontSize: "0.9rem", textTransform: "uppercase", color: "#94a3b8" }}>BERTScore (Similarity)</h3>
              <div style={{ fontSize: "2.5rem", fontWeight: "bold", color: "#818cf8", marginTop: "10px" }}>
                {compLoading ? "..." : similarity ? `${(similarity.f1 * 100).toFixed(1)}%` : "N/A"}
              </div>
            </div>
          </div>

          {similarity && (
            <div style={{ marginTop: "20px", display: "flex", justifyContent: "center", gap: "30px", color: "#94a3b8", fontSize: "0.9rem" }}>
              <span>Precision: {(similarity.precision * 100).toFixed(1)}%</span>
              <span>Recall: {(similarity.recall * 100).toFixed(1)}%</span>
              <span>F1-Score: {(similarity.f1 * 100).toFixed(1)}%</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default App;
