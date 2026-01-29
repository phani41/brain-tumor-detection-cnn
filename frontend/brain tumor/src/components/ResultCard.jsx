export default function ResultCard({ title, data }) {
  if (!data) return null;

  return (
    <div style={{ border: "1px solid #ddd", padding: 12 }}>
      <h3>{title}</h3>
      <p><b>Prediction:</b> {data.prediction}</p>
      <p><b>Confidence:</b> {data.confidence}%</p>
    </div>
  );
}
