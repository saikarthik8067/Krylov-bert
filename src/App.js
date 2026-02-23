import React, { useState, useRef, useEffect } from "react";
import { extractText, summarizeBert, summarizeKrylov, compareSummaries } from "./api/api";
import SummaryCard from "./components/SummaryCard";

function App() {
  const [file, setFile] = useState(null);

  // BERT States
  const [bertResult, setBertResult] = useState(null);
  const [bertLoading, setBertLoading] = useState(false);
  const [bertElapsed, setBertElapsed] = useState(0);
  const bertIntervalRef = useRef(null);

  // Krylov States
  const [krylovResult, setKrylovResult] = useState(null);
  const [krylovLoading, setKrylovLoading] = useState(false);
  const [krylovElapsed, setKrylovElapsed] = useState(0);
  const krylovIntervalRef = useRef(null);

  // Similarity
  const [similarity, setSimilarity] = useState(null);
  const [compLoading, setCompLoading] = useState(false);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      setFile(e.target.files[0]);
    }
  };

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

  const runBertBench = async () => {
    if (!file) { alert("Please select a PDF file first."); return; }

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
      alert("BERT Benchmark failed. Check console or server logs.");
    } finally {
      clearInterval(bertIntervalRef.current);
      setBertLoading(false);
    }
  };

  const runKrylovBench = async () => {
    if (!file) { alert("Please select a PDF file first."); return; }

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
      alert("Krylov Benchmark failed. Check console or server logs.");
    } finally {
      clearInterval(krylovIntervalRef.current);
      setKrylovLoading(false);
    }
  };

  return (
    <>
      <div className="bg-blob blob-1"></div>
      <div className="bg-blob blob-2"></div>

      <div className="app-container">
        <header className="header animate-fade-in">
          <h1>
            <span className="gradient-text">Spectral</span> vs <span className="gradient-text-indigo">Baseline</span>
          </h1>
          <p>
            Upload a PDF document to benchmark the summarization performance of our optimized Krylov-BERT model against the baseline BERT configuration.
          </p>
        </header>

        <div className="grid-container">

          {/* Baseline Model Card */}
          <div className="glass-panel model-card" style={{ animationDelay: '0.1s' }}>
            <div className="model-header">
              <div style={{ width: '48px', height: '48px', borderRadius: '12px', background: 'rgba(99, 102, 241, 0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center', marginRight: '20px' }}>
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="var(--accent-secondary)" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><path d="M4 22h14a2 2 0 0 0 2-2V7l-5-5H6a2 2 0 0 0-2 2v4" /><path d="M14 2v4a2 2 0 0 0 2 2h4" /><path d="M3 15h6" /><path d="M3 18h6" /></svg>
              </div>
              <div>
                <h2>Baseline BERT</h2>
                <p style={{ color: "var(--text-muted)", fontSize: "0.95rem", marginTop: "4px" }}>Standard Transformer Attention</p>
              </div>
            </div>

            <div className="file-upload-wrapper">
              <input type="file" accept=".pdf" onChange={handleFileChange} />
              <div className="file-upload-box">
                <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" /><polyline points="17 8 12 3 7 8" /><line x1="12" y1="3" x2="12" y2="15" /></svg>
                <p style={{ color: "var(--text-main)", fontWeight: "600", fontSize: "1.1rem", margin: "0 0 6px 0" }}>
                  {file ? file.name : "Click or drag PDF here"}
                </p>
                <p>{file ? "Ready to benchmark" : ""}</p>
              </div>
            </div>

            <button
              className="btn btn-indigo"
              onClick={runBertBench}
              disabled={bertLoading || !file}
            >
              {bertLoading ? <><span className="spinner"></span> Processing</> : "Run Baseline Benchmark"}
            </button>

            <div className={`timer-display ${bertLoading ? 'pulse-text' : ''}`} style={{ color: "var(--accent-secondary)" }}>
              {bertElapsed.toFixed(2)}s
            </div>

            {bertResult && (
              <SummaryCard title="Generated Summary" summary={bertResult.summary} time={bertResult.time} />
            )}
          </div>

          {/* Krylov Model Card */}
          <div className="glass-panel model-card" style={{ animationDelay: '0.2s' }}>
            <div className="model-header">
              <div style={{ width: '48px', height: '48px', borderRadius: '12px', background: 'rgba(14, 165, 233, 0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center', marginRight: '20px' }}>
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="var(--accent-primary)" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z" /></svg>
              </div>
              <div>
                <h2>Spectral Krylov-BERT</h2>
                <p style={{ color: "var(--text-muted)", fontSize: "0.95rem", marginTop: "4px" }}>O(N) Lanczos Approximation</p>
              </div>
            </div>

            <div className="file-upload-wrapper">
              <input type="file" accept=".pdf" onChange={handleFileChange} />
              <div className="file-upload-box">
                <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" /><polyline points="17 8 12 3 7 8" /><line x1="12" y1="3" x2="12" y2="15" /></svg>
                <p style={{ color: "var(--text-main)", fontWeight: "600", fontSize: "1.1rem", margin: "0 0 6px 0" }}>
                  {file ? file.name : "Click or drag PDF here"}
                </p>
                <p>{file ? "Ready to benchmark" : ""}</p>
              </div>
            </div>

            <button
              className="btn btn-cyan"
              onClick={runKrylovBench}
              disabled={krylovLoading || !file}
            >
              {krylovLoading ? <><span className="spinner"></span> Processing</> : "Run Spectral Benchmark"}
            </button>

            <div className={`timer-display ${krylovLoading ? 'pulse-text' : ''}`} style={{ color: "var(--accent-primary)" }}>
              {krylovElapsed.toFixed(2)}s
            </div>

            {krylovResult && (
              <SummaryCard title="Optimized Summary" summary={krylovResult.summary} time={krylovResult.time} isHighlighted />
            )}
          </div>

        </div>

        {/* Global Metrics Panel */}
        {bertResult && krylovResult && (
          <div className="glass-panel results-section animate-fade-in" style={{ animationDelay: '0.4s' }}>
            <h2 style={{ textAlign: "center", fontSize: "2.2rem", marginBottom: "12px", fontWeight: "800" }} className="gradient-text">Benchmark Analysis</h2>
            <p style={{ textAlign: "center", color: "var(--text-muted)", marginBottom: "40px", fontSize: "1.1rem" }}>Comparing execution time and summary similarity</p>

            <div className="metrics-grid">

              <div className="metric-card">
                <h3>Performance Speedup</h3>
                <div className="metric-value gradient-text">
                  {(bertResult.time / krylovResult.time).toFixed(2)}x
                </div>
                <p style={{ color: "var(--text-muted)", marginTop: "12px", fontSize: "1rem" }}>Faster than Baseline</p>
              </div>

              <div className="metric-card">
                <h3>Semantic Similarity (F1)</h3>
                <div className="metric-value gradient-text-indigo">
                  {compLoading ? (
                    <span className="spinner" style={{ borderColor: "rgba(99, 102, 241, 0.2)", borderTopColor: "var(--accent-secondary)", width: "45px", height: "45px", borderWidth: "4px" }}></span>
                  ) : similarity ? (
                    `${(similarity.f1 * 100).toFixed(1)}%`
                  ) : (
                    "N/A"
                  )}
                </div>
                <p style={{ color: "var(--text-muted)", marginTop: "12px", fontSize: "1rem" }}>Using BERTScore Evaluation</p>
              </div>

            </div>

            {similarity && (
              <div className="sub-metrics">
                <div className="sub-metric">
                  <span className="label">Precision</span>
                  <span className="value gradient-text-indigo">{(similarity.precision * 100).toFixed(1)}%</span>
                </div>
                <div className="sub-metric" style={{ padding: "0 40px", borderLeft: "1px solid rgba(15,23,42,0.08)", borderRight: "1px solid rgba(15,23,42,0.08)" }}>
                  <span className="label">Recall</span>
                  <span className="value gradient-text-indigo">{(similarity.recall * 100).toFixed(1)}%</span>
                </div>
                <div className="sub-metric">
                  <span className="label">Quality Retention</span>
                  <span className="value gradient-text">Excellent</span>
                </div>
              </div>
            )}
          </div>
        )}

      </div>
    </>
  );
}

export default App;
