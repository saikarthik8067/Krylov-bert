import React from 'react';

const SummaryCard = ({ title, summary, time, isHighlighted }) => {
    return (
        <div className={`glass-panel animate-fade-in ${isHighlighted ? 'highlighted-card' : ''}`} style={{ marginTop: "32px", overflow: "hidden", border: isHighlighted ? "1px solid rgba(14, 165, 233, 0.3)" : "1px solid rgba(15, 23, 42, 0.08)" }}>
            <div style={{
                padding: "20px 28px",
                borderBottom: "1px solid rgba(15,23,42,0.06)",
                background: isHighlighted ? "linear-gradient(90deg, rgba(14, 165, 233, 0.05) 0%, rgba(45, 212, 191, 0.05) 100%)" : "rgba(248, 250, 252, 0.5)",
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center"
            }}>
                <h3 style={{ fontSize: "1.2rem", fontWeight: "700", color: "var(--text-main)", margin: 0 }}>
                    {title}
                </h3>
                {time && (
                    <span style={{
                        fontSize: "0.85rem",
                        fontWeight: "600",
                        padding: "6px 14px",
                        background: "white",
                        border: "1px solid rgba(15,23,42,0.05)",
                        borderRadius: "20px",
                        color: "var(--accent-primary)",
                        boxShadow: "0 2px 5px rgba(0,0,0,0.02)"
                    }}>
                        Processed in {time.toFixed(2)}s
                    </span>
                )}
            </div>
            <div style={{ padding: "28px", fontSize: "1.1rem", lineHeight: "1.8", color: "#334155" }}>
                {summary ? (
                    <p>{summary}</p>
                ) : (
                    <p style={{ color: "var(--text-muted)", fontStyle: "italic" }}>No summary generated yet.</p>
                )}
            </div>
        </div>
    );
};

export default SummaryCard;
