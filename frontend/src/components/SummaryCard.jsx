export default function SummaryCard({ title, time, summary, variant = "bert" }) {
    return (
        <div className="output-box">
            <div className={`output-topbar ${variant}`}>
                <span>⬡ {title}</span>
                <span className={`output-time ${variant}`}>{time?.toFixed(3)}s</span>
            </div>
            <div className="output-text">{summary}</div>
        </div>
    );
}
