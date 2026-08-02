import { useEffect, useMemo, useState } from "react";
import Papa from "papaparse";
import "./index.css";

const metricLabels = {
  faithfulness: "Faithfulness",
  answer_relevancy: "Answer Relevancy",
  context_precision: "Context Precision",
  context_recall: "Context Recall",
  answer_correctness: "Answer Correctness",
};

function toScore(value) {
  const score = Number(value);
  return Number.isFinite(score) ? score : null;
}

function scoreClass(score) {
  if (score === null) return "score-neutral";
  if (score >= 0.8) return "score-good";
  if (score >= 0.5) return "score-warning";
  return "score-bad";
}

function formatContext(value) {
  if (!value) return "No retrieved context available.";

  return String(value)
    .replace(/^\['/, "")
    .replace(/'\]$/, "")
    .replace(/\\n/g, "\n");
}

export default function App() {
  const [rows, setRows] = useState([]);
  const [selectedRow, setSelectedRow] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    fetch("/results/ragas_results.csv")
      .then((response) => {
        if (!response.ok) {
          throw new Error("RAGAS results CSV was not found.");
        }

        return response.text();
      })
      .then((csvText) => {
        const parsed = Papa.parse(csvText, {
          header: true,
          skipEmptyLines: true,
        });

        const normalizedRows = parsed.data.map((row, index) => ({
          ...row,

          id: row.id || `Case ${index + 1}`,
          question: row.question || row.user_input || "",
          answer: row.answer || row.response || "",
          ground_truth: row.ground_truth || row.reference || "",
          contexts: row.contexts || row.retrieved_contexts || "",

          faithfulness: row.faithfulness,
          answer_relevancy:
            row.answer_relevancy ?? row["answer relevancy"],
          context_precision:
            row.context_precision ?? row["context precision"],
          context_recall:
            row.context_recall ?? row["context recall"],
          answer_correctness:
            row.answer_correctness ?? row["answer correctness"],
        }));

        setRows(normalizedRows);
        setSelectedRow(normalizedRows[0] || null);
      })
      .catch((err) => setError(err.message));
  }, []);

  const averages = useMemo(() => {
    return Object.keys(metricLabels).map((metric) => {
      const scores = rows
        .map((row) => toScore(row[metric]))
        .filter((score) => score !== null);

      const average = scores.length
        ? scores.reduce((total, score) => total + score, 0) / scores.length
        : null;

      return { metric, average };
    });
  }, [rows]);

  if (error) {
    return (
      <main className="page">
        <h1>TestCase Compass — RAGAS Dashboard</h1>

        <div className="error">
          <p>{error}</p>
          <p>Run RAGAS first, then copy the CSV into:</p>
          <code>ragas-dashboard/public/results/ragas_results.csv</code>
        </div>
      </main>
    );
  }

  return (
    <main className="page">
      <header>
        <p className="eyebrow">Advanced RAG Evaluation</p>
        <h1>TestCase Compass — RAGAS Dashboard</h1>
        <p className="subtitle">
          Evaluation results for your test-case retrieval assistant.
        </p>
      </header>

      <section className="score-grid">
        {averages.map(({ metric, average }) => (
          <article className="score-card" key={metric}>
            <p>{metricLabels[metric]}</p>
            <strong className={scoreClass(average)}>
              {average === null ? "N/A" : average.toFixed(3)}
            </strong>
          </article>
        ))}
      </section>

      <section className="panel">
        <div className="panel-heading">
          <h2>Evaluation Cases</h2>
          <span>{rows.length} cases</span>
        </div>

        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Test ID</th>
                <th>Question</th>
                <th>Faithfulness</th>
                <th>Correctness</th>
                <th>Precision</th>
                <th>Recall</th>
              </tr>
            </thead>

            <tbody>
              {rows.map((row, index) => (
                <tr
                  key={row.id || index}
                  onClick={() => setSelectedRow(row)}
                  className={selectedRow === row ? "selected" : ""}
                >
                  <td>{row.id}</td>
                  <td>{row.question}</td>

                  <td className={scoreClass(toScore(row.faithfulness))}>
                    {toScore(row.faithfulness)?.toFixed(3) ?? "N/A"}
                  </td>

                  <td className={scoreClass(toScore(row.answer_correctness))}>
                    {toScore(row.answer_correctness)?.toFixed(3) ?? "N/A"}
                  </td>

                  <td className={scoreClass(toScore(row.context_precision))}>
                    {toScore(row.context_precision)?.toFixed(3) ?? "N/A"}
                  </td>

                  <td className={scoreClass(toScore(row.context_recall))}>
                    {toScore(row.context_recall)?.toFixed(3) ?? "N/A"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {selectedRow && (
        <section className="detail-grid">
          <article className="panel">
            <h2>Question</h2>
            <p>{selectedRow.question}</p>

            <h2>Generated Answer</h2>
            <pre>{selectedRow.answer}</pre>
          </article>

          <article className="panel">
            <h2>Reference Answer</h2>
            <pre>{selectedRow.ground_truth}</pre>

            <h2>Retrieved Context</h2>
            <pre>{formatContext(selectedRow.contexts)}</pre>
          </article>
        </section>
      )}
    </main>
  );
}