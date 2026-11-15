import React, { useMemo, useRef, useState } from 'react';
import {
  AlertTriangle,
  CheckCircle2,
  Download,
  FileSpreadsheet,
  Play,
  Upload,
} from 'lucide-react';
import type { BatchRowResult, ModelSchema } from '../types/weather';
import { generateSampleBatchCsv } from '../data/sampleModel';

interface BatchCsvViewProps {
  schema: ModelSchema | null;
  sessionId: string | null;
  onLoadSample: () => void;
}

interface ParsedCsv {
  headers: string[];
  rows: Array<Record<string, number>>;
  parseErrors: string[];
}

function parseCsvText(text: string): ParsedCsv {
  const lines = text
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter((l) => l.length > 0);

  if (lines.length < 2) {
    return {
      headers: [],
      rows: [],
      parseErrors: ['CSV must include a header row and at least one data row.'],
    };
  }

  const headers = lines[0].split(',').map((h) => h.trim().replace(/^"|"$/g, ''));
  const rows: Array<Record<string, number>> = [];
  const parseErrors: string[] = [];

  for (let i = 1; i < lines.length; i++) {
    const cells = lines[i].split(',').map((c) => c.trim().replace(/^"|"$/g, ''));
    const record: Record<string, number> = {};
    let rowInvalid = false;

    headers.forEach((h, idx) => {
      const rawVal = cells[idx] ?? '';
      const num = Number(rawVal);
      if (rawVal === '' || Number.isNaN(num)) {
        rowInvalid = true;
      } else {
        record[h] = num;
      }
    });

    if (rowInvalid) {
      parseErrors.push(`Row ${i} contains missing or non-numeric cells and was skipped.`);
    } else {
      rows.push(record);
    }
  }

  return { headers, rows, parseErrors };
}

function downloadTextFile(filename: string, content: string, mime = 'text/csv') {
  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

export const BatchCsvView: React.FC<BatchCsvViewProps> = ({
  schema,
  sessionId,
  onLoadSample,
}) => {
  const [fileName, setFileName] = useState<string | null>(null);
  const [parsed, setParsed] = useState<ParsedCsv | null>(null);
  const [results, setResults] = useState<BatchRowResult[]>([]);
  const [inferenceMode, setInferenceMode] = useState<string | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [runError, setRunError] = useState<string | null>(null);
  const [classFilter, setClassFilter] = useState<string>('ALL');
  const [selectedRow, setSelectedRow] = useState<BatchRowResult | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const missingColumns = useMemo(() => {
    if (!schema || !parsed) return [];
    const headerSet = new Set(parsed.headers);
    return schema.features.filter((f) => !headerSet.has(f.key)).map((f) => f.key);
  }, [schema, parsed]);

  const classCounts = useMemo(() => {
    const counts = new Map<string, number>();
    for (const r of results) {
      counts.set(r.predictedLabel, (counts.get(r.predictedLabel) || 0) + 1);
    }
    return Array.from(counts.entries());
  }, [results]);

  const filteredResults = useMemo(() => {
    if (classFilter === 'ALL') return results;
    return results.filter((r) => r.predictedLabel === classFilter);
  }, [results, classFilter]);

  if (!schema) {
    return (
      <div className="mx-auto max-w-[1440px] px-6 py-16">
        <div className="border border-slate-800 bg-[#111827] rounded-lg p-10 text-center space-y-4">
          <h2 className="text-xl font-semibold text-slate-100">
            Batch CSV Analyzer Requires an Active Model Schema
          </h2>
          <p className="mx-auto max-w-lg text-sm text-slate-400">
            Inspect your `.pkl` files or load the sample meteorological model first so we know which CSV columns your model expects.
          </p>
          <button
            type="button"
            onClick={onLoadSample}
            className="px-5 py-2.5 text-sm font-semibold text-slate-950 bg-cyan-400 hover:bg-cyan-300 rounded-md transition-colors cursor-pointer"
          >
            Load Sample Weather Model
          </button>
        </div>
      </div>
    );
  }

  const executeBatch = async (rowsToRun: Array<Record<string, number>>) => {
    if (rowsToRun.length === 0) return;
    setIsRunning(true);
    setRunError(null);
    setSelectedRow(null);
    setClassFilter('ALL');

    try {
      const response = await fetch('/api/predict-batch', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sessionId,
          featureOrder: schema.features.map((f) => f.key),
          classes: schema.classes,
          rows: rowsToRun,
        }),
      });
      const data = (await response.json()) as {
        mode?: string;
        results?: BatchRowResult[];
        error?: string;
      };
      if (!response.ok || data.error) {
        throw new Error(data.error || 'Batch prediction failed');
      }
      setInferenceMode(data.mode || 'calibrated-signal-engine');
      setResults(data.results || []);
      if (data.results && data.results.length > 0) {
        setSelectedRow(data.results[0]);
      }
    } catch (err) {
      setRunError(err instanceof Error ? err.message : 'Batch prediction failed');
    } finally {
      setIsRunning(false);
    }
  };

  const handleCsvText = async (name: string, content: string, autoRun = true) => {
    setFileName(name);
    const nextParsed = parseCsvText(content);
    setParsed(nextParsed);
    setResults([]);
    setSelectedRow(null);

    const headerSet = new Set(nextParsed.headers);
    const missing = schema.features.filter((f) => !headerSet.has(f.key));
    if (autoRun && missing.length === 0 && nextParsed.rows.length > 0) {
      await executeBatch(nextParsed.rows);
    }
  };

  const handleFileChange = async (file: File | undefined) => {
    if (!file) return;
    const text = await file.text();
    await handleCsvText(file.name, text, true);
  };

  const handleRunSampleBatch = async () => {
    const csv = generateSampleBatchCsv(schema.features);
    await handleCsvText('sample_weather_observations.csv', csv, true);
  };

  const handleDownloadTemplate = () => {
    const csv = generateSampleBatchCsv(schema.features);
    downloadTextFile('weather_signal_template.csv', csv);
  };

  const handleExportResults = () => {
    if (results.length === 0) return;
    const featureKeys = schema.features.map((f) => f.key);
    const header = [...featureKeys, 'predicted_weather_signal', 'confidence_pct'].join(',');
    const lines = results.map((row) => {
      const vals = featureKeys.map((k) => row.inputs[k] ?? '');
      return [...vals, row.predictedLabel, (row.confidence * 100).toFixed(2)].join(',');
    });
    downloadTextFile('weather_batch_predictions.csv', [header, ...lines].join('\n'));
  };

  return (
    <div className="mx-auto max-w-[1440px] px-6 py-8 space-y-6">
      {/* Header + Actions */}
      <section className="flex flex-col lg:flex-row lg:items-end justify-between gap-4 border-b border-slate-800 pb-6">
        <div className="space-y-1.5">
          <p className="text-xs font-mono uppercase tracking-wider text-cyan-400">
            Batch Telemetry Processing
          </p>
          <h1 className="text-2xl sm:text-3xl font-semibold tracking-tight text-slate-50">
            CSV Batch Forecast Analyzer
          </h1>
          <p className="text-sm text-slate-400">
            Upload a CSV matching your <span className="font-mono text-slate-200">{schema.features.length}</span> schema columns to generate row-by-row weather classifications and confidence scores.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-3 shrink-0">
          <button
            type="button"
            onClick={handleDownloadTemplate}
            className="inline-flex items-center gap-2 px-3.5 py-2 text-xs font-medium text-slate-200 bg-[#111827] hover:bg-slate-800 border border-slate-700 rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer"
          >
            <Download className="w-3.5 h-3.5" />
            <span>Download Template CSV</span>
          </button>
          <button
            type="button"
            onClick={handleRunSampleBatch}
            disabled={isRunning}
            className="inline-flex items-center gap-2 px-3.5 py-2 text-xs font-medium text-slate-200 bg-[#111827] hover:bg-slate-800 border border-slate-700 rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer"
          >
            <Play className="w-3.5 h-3.5 text-cyan-400" />
            <span>Run Sample Batch</span>
          </button>
          {results.length > 0 && (
            <button
              type="button"
              onClick={handleExportResults}
              className="inline-flex items-center gap-2 px-4 py-2 text-xs font-semibold text-slate-950 bg-cyan-400 hover:bg-cyan-300 rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer"
            >
              <Download className="w-3.5 h-3.5" />
              <span>Export Predictions CSV</span>
            </button>
          )}
        </div>
      </section>

      {/* Dropzone */}
      <section
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => {
          e.preventDefault();
          handleFileChange(e.dataTransfer.files?.[0]);
        }}
        className="border border-dashed border-slate-700 bg-[#111827] rounded-lg px-6 py-5 flex flex-col sm:flex-row sm:items-center justify-between gap-4"
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".csv,text/csv"
          className="hidden"
          onChange={(e) => handleFileChange(e.target.files?.[0])}
        />

        <div className="flex items-start sm:items-center gap-3.5">
          <FileSpreadsheet className="w-5 h-5 text-cyan-400 shrink-0 mt-0.5 sm:mt-0" />
          <div className="space-y-0.5">
            <p className="text-sm font-medium text-slate-100">
              {fileName ? fileName : 'Drop weather telemetry CSV here'}
            </p>
            <p className="font-mono text-xs text-slate-400">
              Expected columns: {schema.features.map((f) => f.key).join(', ')}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3 shrink-0">
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            className="inline-flex items-center gap-2 px-4 py-2 text-xs font-medium text-slate-100 bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer"
          >
            <Upload className="w-3.5 h-3.5" />
            <span>Select CSV</span>
          </button>
          {parsed && missingColumns.length === 0 && parsed.rows.length > 0 && (
            <button
              type="button"
              disabled={isRunning}
              onClick={() => executeBatch(parsed.rows)}
              className="inline-flex items-center gap-2 px-4 py-2 text-xs font-semibold text-slate-950 bg-cyan-400 hover:bg-cyan-300 rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer"
            >
              <Play className="w-3.5 h-3.5" />
              <span>{isRunning ? 'Running...' : 'Run Inference'}</span>
            </button>
          )}
        </div>
      </section>

      {/* Validation Status */}
      {parsed && (
        <div className="space-y-2">
          {missingColumns.length > 0 ? (
            <div className="flex items-start gap-2.5 border border-rose-500/40 bg-rose-950/30 px-4 py-3 rounded-lg text-xs text-rose-200">
              <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
              <span>
                Missing required columns: <strong className="font-mono">{missingColumns.join(', ')}</strong>. Download the schema template CSV to inspect exact column names.
              </span>
            </div>
          ) : (
            <div className="flex items-center gap-2.5 border border-emerald-500/30 bg-emerald-950/20 px-4 py-2.5 rounded-lg text-xs text-emerald-200">
              <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
              <span className="font-mono tabular-nums">
                Validated {parsed.rows.length} rows against {schema.features.length} schema features
                {inferenceMode ? ` · Engine: ${inferenceMode}` : ''}
              </span>
            </div>
          )}

          {parsed.parseErrors.map((err, i) => (
            <div
              key={i}
              className="flex items-start gap-2.5 border border-amber-500/30 bg-amber-950/20 px-4 py-2.5 rounded-lg text-xs text-amber-200"
            >
              <AlertTriangle className="w-3.5 h-3.5 text-amber-400 shrink-0 mt-0.5" />
              <span>{err}</span>
            </div>
          ))}
        </div>
      )}

      {runError && (
        <div className="flex items-start gap-2.5 border border-rose-500/40 bg-rose-950/30 px-4 py-3 rounded-lg text-xs text-rose-200">
          <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
          <span>{runError}</span>
        </div>
      )}

      {/* Results Table & Probability Inspector */}
      {results.length > 0 ? (
        <section className="space-y-4">
          {/* Distribution Filter Strip */}
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div className="flex flex-wrap items-center gap-1.5 bg-[#111827] border border-slate-800 p-1 rounded-md">
              <button
                type="button"
                onClick={() => setClassFilter('ALL')}
                className={`px-3 py-1.5 text-xs font-mono rounded transition-colors cursor-pointer whitespace-nowrap shrink-0 ${
                  classFilter === 'ALL'
                    ? 'bg-cyan-400 text-slate-950 font-semibold'
                    : 'text-slate-400 hover:text-slate-100'
                }`}
              >
                All ({results.length})
              </button>
              {classCounts.map(([cls, count]) => (
                <button
                  key={cls}
                  type="button"
                  onClick={() => setClassFilter(cls)}
                  className={`px-3 py-1.5 text-xs font-mono rounded transition-colors cursor-pointer whitespace-nowrap shrink-0 tabular-nums ${
                    classFilter === cls
                      ? 'bg-cyan-400 text-slate-950 font-semibold'
                      : 'text-slate-400 hover:text-slate-100'
                  }`}
                >
                  {cls} ({count})
                </button>
              ))}
            </div>
          </div>

          <div className="grid grid-cols-1 xl:grid-cols-12 gap-6 items-start">
            {/* Table */}
            <div className="xl:col-span-8 border border-slate-800 bg-[#111827] rounded-lg overflow-hidden">
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead className="border-b border-slate-800 bg-[#0B0E17] text-slate-400 font-mono">
                    <tr>
                      <th className="px-4 py-3 font-medium">#</th>
                      <th className="px-4 py-3 font-medium">Predicted Signal</th>
                      <th className="px-4 py-3 font-medium text-right">Confidence</th>
                      {schema.features.slice(0, 4).map((f) => (
                        <th key={f.key} className="px-4 py-3 font-medium text-right">
                          {f.key}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/80 font-mono tabular-nums">
                    {filteredResults.map((row) => {
                      const isSelected = selectedRow?.rowIndex === row.rowIndex;
                      return (
                        <tr
                          key={row.rowIndex}
                          onClick={() => setSelectedRow(row)}
                          className={`cursor-pointer transition-colors ${
                            isSelected ? 'bg-cyan-950/35' : 'hover:bg-slate-800/50'
                          }`}
                        >
                          <td className="px-4 py-2.5 text-slate-500">{row.rowIndex}</td>
                          <td className="px-4 py-2.5 text-slate-100 font-semibold">
                            {row.predictedLabel}
                          </td>
                          <td className="px-4 py-2.5 text-right text-cyan-300">
                            {(row.confidence * 100).toFixed(1)}%
                          </td>
                          {schema.features.slice(0, 4).map((f) => (
                            <td key={f.key} className="px-4 py-2.5 text-right text-slate-300">
                              {Number(row.inputs[f.key] ?? 0).toFixed(1)}
                            </td>
                          ))}
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Selected Row Breakdown */}
            <aside className="xl:col-span-4 border border-slate-800 bg-[#111827] rounded-lg p-5 space-y-4">
              {selectedRow ? (
                <>
                  <div className="border-b border-slate-800 pb-3">
                    <p className="text-xs font-mono text-slate-400">
                      Observation Row #{selectedRow.rowIndex}
                    </p>
                    <div className="mt-1 flex items-baseline justify-between">
                      <h3 className="text-xl font-semibold text-slate-50">
                        {selectedRow.predictedLabel}
                      </h3>
                      <span className="font-mono text-sm font-semibold text-cyan-300 tabular-nums">
                        {(selectedRow.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>

                  <div className="space-y-2.5">
                    <p className="text-xs font-medium text-slate-400">
                      Class Probabilities
                    </p>
                    {selectedRow.probabilities.map((p) => (
                      <div key={p.label} className="space-y-1">
                        <div className="flex justify-between font-mono text-xs tabular-nums">
                          <span className="text-slate-300">{p.label}</span>
                          <span className="text-slate-200">
                            {(p.probability * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div className="h-1.5 bg-[#0B0E17] rounded overflow-hidden">
                          <div
                            className="h-full bg-cyan-400"
                            style={{ width: `${p.probability * 100}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>

                  <div className="border-t border-slate-800 pt-3 space-y-1.5">
                    <p className="text-xs font-medium text-slate-400">
                      Raw Feature Vector
                    </p>
                    <div className="grid grid-cols-2 gap-x-4 gap-y-1 font-mono text-xs tabular-nums">
                      {schema.features.map((f) => (
                        <div key={f.key} className="flex justify-between gap-2">
                          <span className="text-slate-500 truncate">{f.key}</span>
                          <span className="text-slate-200">
                            {selectedRow.inputs[f.key]}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                </>
              ) : (
                <p className="text-xs text-slate-400">
                  Select a row from the batch table to inspect its class probabilities and feature vector.
                </p>
              )}
            </aside>
          </div>
        </section>
      ) : (
        <section className="border border-slate-800 bg-[#111827] rounded-lg p-8 text-center space-y-3">
          <p className="text-sm font-medium text-slate-200">
            Ready for Batch Inference
          </p>
          <p className="text-xs text-slate-400 max-w-md mx-auto">
            Click &ldquo;Run Sample Batch&rdquo; to test 5 realistic meteorological observations immediately, or upload your own CSV file.
          </p>
        </section>
      )}
    </div>
  );
};
