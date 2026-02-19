export default function SummaryCard({ title, time, summary, highlight }) {
    return (
        <div style={{
            backgroundColor: highlight ? "#0ea5e9" : "#1e293b",
            padding: "20px",
            borderRadius: "12px",
            flex: 1,
            color: "white"
        }}>
            <h3>{title}</h3>
            <p><strong>Time:</strong> {time?.toFixed(2)} sec</p>
            <p>{summary}</p>
        </div>
    );
}
