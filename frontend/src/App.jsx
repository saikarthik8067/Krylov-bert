import React, { useState, useRef, useEffect } from "react";
import "./App.css";
import { extractText, summarizeBert, summarizeKrylov, compareSummaries } from "./api/api";
import SummaryCard from "./components/SummaryCard";

/* ── Helpers ── */
const fmt  = (n) => n.toFixed(2);
const pct  = (n) => `${(n * 100).toFixed(1)}%`;

function App() {
    const [file,         setFile]         = useState(null);
    const [drag,         setDrag]         = useState(false);

    const [bertResult,   setBertResult]   = useState(null);
    const [bertLoading,  setBertLoading]  = useState(false);
    const [bertElapsed,  setBertElapsed]  = useState(0);
    const bertTimer = useRef(null);

    const [kResult,      setKResult]      = useState(null);
    const [kLoading,     setKLoading]     = useState(false);
    const [kElapsed,     setKElapsed]     = useState(0);
    const kTimer = useRef(null);

    const [similarity,   setSimilarity]   = useState(null);
    const [compLoading,  setCompLoading]  = useState(false);

    /* Auto compare */
    useEffect(() => {
        if (bertResult && kResult) {
            setCompLoading(true);
            compareSummaries(kResult.summary, bertResult.summary)
                .then(setSimilarity)
                .catch(console.error)
                .finally(() => setCompLoading(false));
        } else {
            setSimilarity(null);
        }
    }, [bertResult, kResult]);

    /* File handlers */
    const onFile = (f) => { if (f?.name.endsWith(".pdf")) setFile(f); };
    const onDrop = (e) => {
        e.preventDefault(); setDrag(false);
        onFile(e.dataTransfer.files[0]);
    };

    /* BERT run */
    const runBert = async () => {
        if (!file) return alert("Upload a PDF first.");
        setBertLoading(true); setBertResult(null); setBertElapsed(0);
        const t0 = Date.now();
        bertTimer.current = setInterval(() => setBertElapsed((Date.now() - t0) / 1000), 40);
        try {
            const text = await extractText(file);
            setBertResult(await summarizeBert(text));
        } catch { alert("BERT failed — is the backend running on :8000?"); }
        finally { clearInterval(bertTimer.current); setBertLoading(false); }
    };

    /* Krylov run */
    const runKrylov = async () => {
        if (!file) return alert("Upload a PDF first.");
        setKLoading(true); setKResult(null); setKElapsed(0);
        const t0 = Date.now();
        kTimer.current = setInterval(() => setKElapsed((Date.now() - t0) / 1000), 40);
        try {
            const text = await extractText(file);
            setKResult(await summarizeKrylov(text));
        } catch { alert("Krylov failed — is the backend running on :8000?"); }
        finally { clearInterval(kTimer.current); setKLoading(false); }
    };

    const busy     = bertLoading || kLoading;
    const bothDone = bertResult && kResult;
    const speedup  = bothDone ? (bertResult.time / kResult.time).toFixed(2) : null;
    const maxTime  = bothDone ? Math.max(bertResult.time, kResult.time) : 1;

    return (
        <div className="app">
            {/* ── Background ── */}
            <div className="bg-grid" />
            <div className="bg-orb bg-orb-1" />
            <div className="bg-orb bg-orb-2" />
            <div className="bg-orb bg-orb-3" />

            {/* ── Navbar ── */}
            <nav className="navbar">
                <div className="nav-inner">
                    <div className="nav-brand">
                        <div className="nav-icon">⚡</div>
                        <span className="nav-wordmark">Krylov<em>BERT</em></span>
                    </div>
                    <div className="nav-right">
                        <span className="nav-pill version">v1.0</span>
                        <span className="nav-pill research">Research</span>
                        <span className="nav-dot" title="Backend connected" />
                    </div>
                </div>
            </nav>

            <div className="page-content">

                {/* ── Hero ── */}
                <section className="hero">
                    <div className="hero-eyebrow">
                        <span className="eyebrow-dot" />
                        Spectral Sequence Modelling · NLP Benchmark
                    </div>
                    <h1 className="hero-h1">
                        <span className="grad-text">Spectral Krylov</span>
                        <br />vs Baseline BERT
                    </h1>
                    <p className="hero-desc">
                        Upload a research PDF and benchmark extractive summarisation
                        performance between the novel Krylov‑subspace architecture
                        and standard BERT — side by side, in real time.
                    </p>
                    <div className="hero-sub">
                        <span className="hero-sub-item">PyTorch backend</span>
                        <span className="hero-sub-item">BERTScore evaluation</span>
                        <span className="hero-sub-item">Live inference timer</span>
                    </div>
                </section>

                {/* ── Upload ── */}
                <div className="upload-wrapper">
                    <div
                        className={`upload-drop ${drag ? "drag" : ""}`}
                        onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
                        onDragLeave={() => setDrag(false)}
                        onDrop={onDrop}
                    >
                        <input
                            type="file"
                            accept=".pdf"
                            id="pdf-input"
                            onChange={(e) => onFile(e.target.files[0])}
                        />
                        <span className="upload-emoji">📄</span>
                        <div className="upload-title">
                            {drag ? "Release to upload" : "Drop your PDF here"}
                        </div>
                        <div className="upload-hint">
                            or <strong>click to browse</strong> · PDF files only
                        </div>
                    </div>

                    {file && (
                        <div className="file-chip">
                            <span className="file-chip-icon">✓</span>
                            <span className="file-chip-name">{file.name}</span>
                            <span className="file-chip-size">
                                {(file.size / 1024).toFixed(1)} KB
                            </span>
                        </div>
                    )}
                </div>

                {/* ── Benchmark columns ── */}
                <div className="bench-area">

                    {/* ── BERT Panel ── */}
                    <div className="model-panel bert">
                        <div className="panel-accent bert" />
                        <div className="panel-body">
                            <div className="panel-header">
                                <div className="panel-avatar bert">🔵</div>
                                <div>
                                    <div className="panel-name bert">Baseline BERT</div>
                                    <div className="panel-meta">bert-base-uncased · CLS pooling</div>
                                </div>
                                <span className="panel-tag bert">Baseline</span>
                            </div>

                            <div className="timer-block">
                                <div className="timer-ring bert">
                                    <span className="timer-num bert">{fmt(bertElapsed)}</span>
                                    <span className="timer-unit">seconds</span>
                                </div>
                            </div>

                            <button
                                id="btn-bert"
                                className="infer-btn bert"
                                onClick={runBert}
                                disabled={busy}
                            >
                                {bertLoading
                                    ? <><div className="spin" /> Computing…</>
                                    : <><span>▶</span> Run BERT Benchmark</>
                                }
                            </button>

                            {bertLoading && (
                                <div className="proc-label bert">
                                    <span className="proc-dot" />
                                    Encoding sentences via CLS pooling…
                                </div>
                            )}

                            {bertResult && (
                                <SummaryCard
                                    title="BERT Output"
                                    summary={bertResult.summary}
                                    time={bertResult.time}
                                    variant="bert"
                                />
                            )}
                        </div>
                    </div>

                    {/* ── VS Divider ── */}
                    <div className="vs-divider">
                        <div className="vs-line" />
                        <div className="vs-badge">VS</div>
                        <div className="vs-line" />
                    </div>

                    {/* ── Krylov Panel ── */}
                    <div className="model-panel krylov">
                        <div className="panel-accent krylov" />
                        <div className="panel-body">
                            <div className="panel-header">
                                <div className="panel-avatar krylov">⚡</div>
                                <div>
                                    <div className="panel-name krylov">Spectral Krylov-BERT</div>
                                    <div className="panel-meta">d_model=128 · Krylov subspace</div>
                                </div>
                                <span className="panel-tag krylov">Proposed</span>
                            </div>

                            <div className="timer-block">
                                <div className="timer-ring krylov">
                                    <span className="timer-num krylov">{fmt(kElapsed)}</span>
                                    <span className="timer-unit">seconds</span>
                                </div>
                            </div>

                            <button
                                id="btn-krylov"
                                className="infer-btn krylov"
                                onClick={runKrylov}
                                disabled={busy}
                            >
                                {kLoading
                                    ? <><div className="spin" style={{ borderTopColor: "#030a0c" }} /> Computing…</>
                                    : <><span>⚡</span> Run Krylov Benchmark</>
                                }
                            </button>

                            {kLoading && (
                                <div className="proc-label krylov">
                                    <span className="proc-dot" />
                                    Running Krylov spectral forward pass…
                                </div>
                            )}

                            {kResult && (
                                <SummaryCard
                                    title="Krylov Output"
                                    summary={kResult.summary}
                                    time={kResult.time}
                                    variant="krylov"
                                />
                            )}
                        </div>
                    </div>

                </div>

                {/* ── Results ── */}
                {bothDone && (
                    <div className="results-section" id="results">

                        <div className="section-divider">Analysis Results</div>

                        <div className="results-heading">
                            <h2>⚡ Benchmark Analysis</h2>
                            <p>Comparing Krylov‑BERT against the BERT baseline on your document</p>
                        </div>

                        {/* Stat cards */}
                        <div className="stats-strip">
                            <div className="stat-card speedup">
                                <div className="stat-icon">🚀</div>
                                <div className="stat-label">Speed Improvement</div>
                                <div className="stat-value cyan">{speedup}×</div>
                                <div className="stat-sub">Krylov vs BERT</div>
                            </div>
                            <div className="stat-card bert-t">
                                <div className="stat-icon">⏱</div>
                                <div className="stat-label">BERT Inference</div>
                                <div className="stat-value violet">{fmt(bertResult.time)}s</div>
                                <div className="stat-sub">Baseline time</div>
                            </div>
                            <div className="stat-card krylov-t">
                                <div className="stat-icon">⚡</div>
                                <div className="stat-label">Krylov Inference</div>
                                <div className="stat-value cyan">{fmt(kResult.time)}s</div>
                                <div className="stat-sub">Proposed time</div>
                            </div>
                            <div className="stat-card f1">
                                <div className="stat-icon">🎯</div>
                                <div className="stat-label">BERTScore F1</div>
                                <div className="stat-value green">
                                    {compLoading ? "…" : similarity ? pct(similarity.f1) : "—"}
                                </div>
                                <div className="stat-sub">Semantic similarity</div>
                            </div>
                        </div>

                        {/* Bar comparison */}
                        <div className="compare-block">
                            <div className="compare-title">⬡ Inference Time Comparison</div>
                            <div className="bar-row">
                                <div className="bar-label">Baseline BERT</div>
                                <div className="bar-track">
                                    <div
                                        className="bar-fill bert"
                                        style={{ "--w": `${Math.min(100, (bertResult.time / maxTime) * 100)}%` }}
                                    />
                                </div>
                                <div className="bar-val">{fmt(bertResult.time)}s</div>
                            </div>
                            <div className="bar-row">
                                <div className="bar-label">Krylov-BERT</div>
                                <div className="bar-track">
                                    <div
                                        className="bar-fill krylov"
                                        style={{ "--w": `${Math.min(100, (kResult.time / maxTime) * 100)}%` }}
                                    />
                                </div>
                                <div className="bar-val">{fmt(kResult.time)}s</div>
                            </div>
                        </div>

                        {/* BERTScore pills */}
                        {similarity && (
                            <div className="bscore-row">
                                <div className="bscore-pill">
                                    <span className="bscore-pill-label">Precision</span>
                                    <span className="bscore-pill-val">{pct(similarity.precision)}</span>
                                </div>
                                <div className="bscore-pill">
                                    <span className="bscore-pill-label">Recall</span>
                                    <span className="bscore-pill-val">{pct(similarity.recall)}</span>
                                </div>
                                <div className="bscore-pill">
                                    <span className="bscore-pill-label">F1 Score</span>
                                    <span className="bscore-pill-val">{pct(similarity.f1)}</span>
                                </div>
                            </div>
                        )}

                    </div>
                )}

            </div>
        </div>
    );
}

export default App;
