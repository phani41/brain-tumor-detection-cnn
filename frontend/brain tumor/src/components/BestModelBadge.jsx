// components/BestModelBadge.jsx
export default function BestModelBadge({ bestModel }) {
  if (!bestModel || typeof bestModel !== "object") return null;

  return (
    <div style={{
      background: "#f5f5f5",
      padding: "16px",
      borderRadius: "8px",
      marginBottom: "20px"
    }}>
      <h3>Best Model Decision</h3>
      <p><strong>Model:</strong> {bestModel.model}</p>
      <p><strong>Prediction:</strong> {bestModel.prediction}</p>
      <p><strong>Confidence:</strong> {bestModel.confidence}%</p>
    </div>
  );
}
