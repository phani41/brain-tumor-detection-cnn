import { Bar } from "react-chartjs-2";
import {
  Chart as ChartJS,
  BarElement,
  CategoryScale,
  LinearScale,
  Tooltip,
  Legend,
} from "chart.js";

ChartJS.register(BarElement, CategoryScale, LinearScale, Tooltip, Legend);

export default function ComparisonCharts({ mobilenet, efficientnet }) {
  if (!mobilenet || !efficientnet) return null;

  const labels = ["glioma", "meningioma", "notumor", "pituitary"];

  return (
    <>
      <Bar
        data={{
          labels,
          datasets: [
            {
              label: "MobileNet",
              data: labels.map(l => mobilenet.probabilities[l]),
              backgroundColor: "#38bdf8",
            },
            {
              label: "EfficientNet",
              data: labels.map(l => efficientnet.probabilities[l]),
              backgroundColor: "#4ade80",
            },
          ],
        }}
      />

      <Bar
        data={{
          labels: ["MobileNet", "EfficientNet"],
          datasets: [
            {
              label: "Confidence (%)",
              data: [mobilenet.confidence, efficientnet.confidence],
              backgroundColor: ["#38bdf8", "#4ade80"],
            },
          ],
        }}
        options={{ scales: { y: { beginAtZero: true, max: 100 } } }}
      />
    </>
  );
}
