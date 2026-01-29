import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer
} from "recharts";

export default function ProbabilityChart({ title, probabilities }) {
  if (!probabilities) return <p>Loading chart...</p>;

  const data = Object.keys(probabilities).map(k => ({
    name: k,
    value: probabilities[k]
  }));

  return (
    <div style={{ background: "#111", padding: 15 }}>
      <h4 style={{ color: "white" }}>{title}</h4>

      <ResponsiveContainer width="100%" height={250}>
        <BarChart data={data}>
          <XAxis dataKey="name" stroke="#fff" />
          <YAxis stroke="#fff" />
          <Tooltip />
          <Bar dataKey="value" fill="#8884d8" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
