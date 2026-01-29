import { useState } from "react";
import UploadImage from "../components/UploadImage";
import ResultCard from "../components/ResultCard";
import BestModelBadge from "../components/BestModelBadge";
import ComparisonCharts from "../components/ComparisonCharts";
import { compareModels } from "../services/api";

export default function ComparePage() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleCompare = async (file) => {
    try {
      setLoading(true);
      setError("");
      setResult(null);

      const response = await compareModels(file);
      setResult(response); // ✅ STORE RESPONSE IN STATE
    } catch (err) {
      setError("Error connecting to backend");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: "20px" }}>
      <h1>Brain Tumor Model Comparison</h1>

      <UploadImage onUpload={handleCompare} />

      {loading && <p>Analyzing image...</p>}
      {error && <p style={{ color: "red" }}>{error}</p>}

      {result && (
        <>
          {/* ✅ FIXED */}
          <BestModelBadge bestModel={result.best_model} />

          <div style={{ display: "flex", gap: "20px" }}>
            <ResultCard title="MobileNet" data={result.mobilenet} />
            <ResultCard title="EfficientNet" data={result.efficientnet} />
          </div>

          <ComparisonCharts
            mobilenet={result.mobilenet}
            efficientnet={result.efficientnet}
          />
        </>
      )}
    </div>
  );
}
