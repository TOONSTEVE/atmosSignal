import React, { useRef, useState } from 'react';
import {
  AlertTriangle,
  ArrowRight,
  CheckCircle2,
  FileCode2,
  Layers,
  Plus,
  Trash2,
  Upload,
} from 'lucide-react';
import type { FeatureSpec, ModelSchema, RuntimeCapabilities } from '../types/weather';
import { inferFeatureSpec } from '../data/sampleModel';

interface InspectorViewProps {
  schema: ModelSchema | null;
  capabilities: RuntimeCapabilities | null;
  isInspecting: boolean;
  inspectError: string | null;
  onInspectFiles: (modelFile: File | null, encoderFile: File | null) => Promise<void>;
  onLoadJsonFile: (jsonFile: File) => Promise<void>;
  onLoadSample: () => void;
  onUpdateSchema: (next: ModelSchema) => void;
  onNavigate: (tab: 'forecast' | 'batch' | 'bridge') => void;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

export const InspectorView: React.FC<InspectorViewProps> = ({
  schema,
  capabilities,
  isInspecting,
  inspectError,
  onInspectFiles,
  onLoadJsonFile,
  onLoadSample,
  onUpdateSchema,
  onNavigate,
}) => {
  const [uploadMode, setUploadMode] = useState<'pkl' | 'json'>('pkl');
  const [modelFile, setModelFile] = useState<File | null>(null);
  const [encoderFile, setEncoderFile] = useState<File | null>(null);
  const [jsonFile, setJsonFile] = useState<File | null>(null);
  const [newFeatureName, setNewFeatureName] = useState('');
  const [newClassName, setNewClassName] = useState('');

  const modelInputRef = useRef<HTMLInputElement>(null);
  const encoderInputRef = useRef<HTMLInputElement>(null);
  const jsonInputRef = useRef<HTMLInputElement>(null);

  const handleModelDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const dropped = e.dataTransfer.files?.[0];
    if (dropped) setModelFile(dropped);
  };

  const handleEncoderDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const dropped = e.dataTransfer.files?.[0];
    if (dropped) setEncoderFile(dropped);
  };

  const handleRunInspection = async () => {
    if (!modelFile && !encoderFile) return;
    await onInspectFiles(modelFile, encoderFile);
  };

  const handleAddFeature = () => {
    if (!schema) return;
    const clean = newFeatureName.trim();
    if (!clean) return;
    if (schema.features.some((f) => f.key === clean)) return;
    const added: FeatureSpec = inferFeatureSpec(clean, schema.features.length);
    onUpdateSchema({
      ...schema,
      features: [...schema.features, added],
    });
    setNewFeatureName('');
  };

  const handleRemoveFeature = (key: string) => {
    if (!schema || schema.features.length <= 1) return;
    onUpdateSchema({
      ...schema,
      features: schema.features.filter((f) => f.key !== key),
    });
  };

  const handleAddClass = () => {
    if (!schema) return;
    const clean = newClassName.trim();
    if (!clean || schema.classes.includes(clean)) return;
    onUpdateSchema({
      ...schema,
      classes: [...schema.classes, clean],
    });
    setNewClassName('');
  };

  const handleRemoveClass = (cls: string) => {
    if (!schema || schema.classes.length <= 1) return;
    onUpdateSchema({
      ...schema,
      classes: schema.classes.filter((c) => c !== cls),
    });
  };

  return (
    <div className="mx-auto max-w-[1440px] px-6 py-8 space-y-8">
      {/* Header Section */}
      <section className="flex flex-col gap-6 lg:flex-row lg:items-end lg:justify-between border-b border-slate-800 pb-6">
        <div className="max-w-3xl space-y-2">
          <p className="text-xs font-mono uppercase tracking-wider text-cyan-400">
            Model Artifact Workbench
          </p>
          <h1 className="text-3xl sm:text-4xl font-semibold tracking-tight text-slate-50 [text-wrap:balance]">
            Inspect .pkl Weather Models and Build Your Forecast Console
          </h1>
          <p className="text-sm sm:text-base text-slate-400 leading-relaxed">
            A <span className="font-mono text-slate-200">.pkl</span> file stores a serialized Python machine learning object created with <span className="font-mono text-slate-200">pickle</span> or <span className="font-mono text-slate-200">joblib</span>. Upload <span className="font-mono text-slate-200">weather_signal_model.pkl</span> and <span className="font-mono text-slate-200">label_encoder.pkl</span> below to extract their estimator type, expected input features, and target weather classes.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-3 shrink-0">
          <button
            type="button"
            onClick={onLoadSample}
            className="px-4 py-2.5 text-sm font-medium text-slate-200 bg-[#111827] hover:bg-slate-800 border border-slate-700 rounded-md transition-colors duration-150 whitespace-nowrap shrink-0 cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-500"
          >
            Load Sample Weather Model
          </button>
          {schema && (
            <button
              type="button"
              onClick={() => onNavigate('forecast')}
              className="inline-flex items-center gap-2 px-4 py-2.5 text-sm font-semibold text-slate-950 bg-cyan-400 hover:bg-cyan-300 rounded-md transition-colors duration-150 whitespace-nowrap shrink-0 cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-300"
            >
              <span>Open Forecast Console</span>
              <ArrowRight className="w-4 h-4" />
            </button>
          )}
        </div>
      </section>

      {/* Runtime Capability Banner (Honest, text-first status) */}
      {capabilities && (
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border border-slate-800 bg-[#111827] px-5 py-3.5 rounded-lg">
          <div className="flex items-start sm:items-center gap-3">
            {capabilities.canRunNativePklInference ? (
              <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5 sm:mt-0" />
            ) : (
              <Layers className="w-4 h-4 text-amber-400 shrink-0 mt-0.5 sm:mt-0" />
            )}
            <div className="text-xs sm:text-sm text-slate-300">
              {capabilities.canRunNativePklInference ? (
                <span>
                  <strong className="font-semibold text-emerald-300">Live Scikit-Learn Runtime Active</strong>
                  <span className="text-slate-500 mx-2">·</span>
                  <span className="font-mono text-xs text-slate-400">
                    Python {capabilities.pythonVersion} · sklearn {capabilities.sklearnVersion}
                  </span>
                </span>
              ) : (
                <span>
                  <strong className="font-semibold text-amber-300">Safe Opcode Schema Inspector Active</strong>
                  <span className="text-slate-500 mx-2">·</span>
                  <span className="text-slate-400">
                    Python {capabilities.pythonVersion} stdlib inspects `.pkl` schemas directly; `scikit-learn` is not installed in this cloud container, so forecasts run via the calibrated signal engine and the Python Bridge tab provides your local FastAPI server script.
                  </span>
                </span>
              )}
            </div>
          </div>
          <button
            type="button"
            onClick={() => onNavigate('bridge')}
            className="text-xs font-medium text-cyan-400 hover:text-cyan-300 whitespace-nowrap shrink-0 cursor-pointer"
          >
            View Python Setup Guide
          </button>
        </div>
      )}

      {/* Mode Selector: .pkl Inspection vs Exported .json Bundle */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div className="inline-flex items-center gap-1 bg-[#111827] border border-slate-800 p-1 rounded-md self-start">
          <button
            type="button"
            onClick={() => setUploadMode('pkl')}
            className={`px-3.5 py-1.5 text-xs font-medium rounded transition-colors cursor-pointer whitespace-nowrap shrink-0 ${
              uploadMode === 'pkl'
                ? 'bg-cyan-400 text-slate-950 font-semibold'
                : 'text-slate-400 hover:text-slate-100'
            }`}
          >
            Inspect .pkl Files
          </button>
          <button
            type="button"
            onClick={() => setUploadMode('json')}
            className={`px-3.5 py-1.5 text-xs font-medium rounded transition-colors cursor-pointer whitespace-nowrap shrink-0 ${
              uploadMode === 'json'
                ? 'bg-cyan-400 text-slate-950 font-semibold'
                : 'text-slate-400 hover:text-slate-100'
            }`}
          >
            Load Exported .json Bundle (Exact Weights)
          </button>
        </div>

        {schema?.source === 'exported-json' && (
          <div className="flex items-center gap-2 text-xs text-emerald-300 font-mono">
            <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
            <span>Exact Exported Model Active (Zero Python Server Needed)</span>
          </div>
        )}
      </div>

      {uploadMode === 'json' ? (
        <section className="space-y-4">
          <div
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => {
              e.preventDefault();
              const dropped = e.dataTransfer.files?.[0];
              if (dropped) setJsonFile(dropped);
            }}
            className="border border-slate-800 bg-[#111827] p-6 rounded-lg space-y-4"
          >
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
              <div>
                <h2 className="text-base font-semibold text-slate-100">
                  Upload Exported Model Bundle (weather_signal_bundle.json)
                </h2>
                <p className="text-xs text-slate-400 mt-1">
                  Run <span className="font-mono text-slate-200">export_to_json.py</span> from the Python Bridge tab once on your machine, then drop the generated <span className="font-mono text-slate-200">.json</span> bundle here to execute your exact trained trees/weights inside the app.
                </p>
              </div>
              <button
                type="button"
                onClick={() => onNavigate('bridge')}
                className="text-xs font-medium text-cyan-400 hover:text-cyan-300 whitespace-nowrap shrink-0 self-start sm:self-center cursor-pointer"
              >
                Get export_to_json.py →
              </button>
            </div>

            <input
              ref={jsonInputRef}
              type="file"
              accept=".json,application/json"
              className="hidden"
              onChange={(e) => setJsonFile(e.target.files?.[0] ?? null)}
            />

            <div className="flex items-center justify-between gap-4 border border-dashed border-slate-700 bg-[#0B0E17] px-4 py-4 rounded-md">
              <div className="min-w-0">
                {jsonFile ? (
                  <div className="space-y-0.5">
                    <p className="font-mono text-xs text-cyan-300 truncate">
                      {jsonFile.name}
                    </p>
                    <p className="font-mono text-xs text-slate-500 tabular-nums">
                      {formatBytes(jsonFile.size)}
                    </p>
                  </div>
                ) : (
                  <p className="text-xs text-slate-500">
                    Drop weather_signal_bundle.json here or select a file
                  </p>
                )}
              </div>
              <button
                type="button"
                onClick={() => jsonInputRef.current?.click()}
                className="inline-flex items-center gap-2 px-3.5 py-2 text-xs font-medium text-slate-100 bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer"
              >
                <Upload className="w-3.5 h-3.5" />
                <span>Select JSON</span>
              </button>
            </div>
          </div>

          <div className="flex flex-wrap items-center justify-between gap-4">
            <div className="text-xs text-slate-400">
              Runs your exact scikit-learn scalers, trees, or weights natively in Express with zero Python dependencies.
            </div>
            <button
              type="button"
              disabled={!jsonFile || isInspecting}
              onClick={() => {
                if (jsonFile) void onLoadJsonFile(jsonFile);
              }}
              className="inline-flex items-center gap-2 px-5 py-2.5 text-sm font-semibold text-slate-950 bg-cyan-400 hover:bg-cyan-300 disabled:bg-slate-800 disabled:text-slate-500 rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer disabled:cursor-not-allowed"
            >
              <FileCode2 className="w-4 h-4" />
              <span>{isInspecting ? 'Loading Bundle...' : 'Load Exported JSON Model'}</span>
            </button>
          </div>
        </section>
      ) : (
        <>
          {/* Paired .pkl Upload Zone */}
          <section className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div
          onDragOver={(e) => e.preventDefault()}
          onDrop={handleModelDrop}
          className="border border-slate-800 bg-[#111827] p-6 rounded-lg flex flex-col justify-between gap-4"
        >
          <div className="space-y-1.5">
            <div className="flex items-center justify-between gap-4">
              <h2 className="text-base font-semibold text-slate-100">
                1. Trained Weather Model
              </h2>
              <span className="font-mono text-xs text-slate-400">
                weather_signal_model.pkl
              </span>
            </div>
            <p className="text-xs text-slate-400">
              Drop your scikit-learn classifier, regressor, or pipeline `.pkl` file here.
            </p>
          </div>

          <input
            ref={modelInputRef}
            type="file"
            accept=".pkl,.pickle,.joblib"
            className="hidden"
            onChange={(e) => setModelFile(e.target.files?.[0] ?? null)}
          />

          <div className="flex items-center justify-between gap-4 border border-dashed border-slate-700 bg-[#0B0E17] px-4 py-4 rounded-md">
            <div className="min-w-0">
              {modelFile ? (
                <div className="space-y-0.5">
                  <p className="font-mono text-xs text-cyan-300 truncate">
                    {modelFile.name}
                  </p>
                  <p className="font-mono text-xs text-slate-500 tabular-nums">
                    {formatBytes(modelFile.size)}
                  </p>
                </div>
              ) : (
                <p className="text-xs text-slate-500">
                  No model file selected yet
                </p>
              )}
            </div>
            <button
              type="button"
              onClick={() => modelInputRef.current?.click()}
              className="inline-flex items-center gap-2 px-3.5 py-2 text-xs font-medium text-slate-100 bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer"
            >
              <Upload className="w-3.5 h-3.5" />
              <span>Select File</span>
            </button>
          </div>
        </div>

        <div
          onDragOver={(e) => e.preventDefault()}
          onDrop={handleEncoderDrop}
          className="border border-slate-800 bg-[#111827] p-6 rounded-lg flex flex-col justify-between gap-4"
        >
          <div className="space-y-1.5">
            <div className="flex items-center justify-between gap-4">
              <h2 className="text-base font-semibold text-slate-100">
                2. Target Label Encoder
              </h2>
              <span className="font-mono text-xs text-slate-400">
                label_encoder.pkl
              </span>
            </div>
            <p className="text-xs text-slate-400">
              Drop your `LabelEncoder` `.pkl` file to decode numeric predictions (0, 1, 2) into weather labels.
            </p>
          </div>

          <input
            ref={encoderInputRef}
            type="file"
            accept=".pkl,.pickle,.joblib"
            className="hidden"
            onChange={(e) => setEncoderFile(e.target.files?.[0] ?? null)}
          />

          <div className="flex items-center justify-between gap-4 border border-dashed border-slate-700 bg-[#0B0E17] px-4 py-4 rounded-md">
            <div className="min-w-0">
              {encoderFile ? (
                <div className="space-y-0.5">
                  <p className="font-mono text-xs text-cyan-300 truncate">
                    {encoderFile.name}
                  </p>
                  <p className="font-mono text-xs text-slate-500 tabular-nums">
                    {formatBytes(encoderFile.size)}
                  </p>
                </div>
              ) : (
                <p className="text-xs text-slate-500">
                  Optional if classes are stored inside the model
                </p>
              )}
            </div>
            <button
              type="button"
              onClick={() => encoderInputRef.current?.click()}
              className="inline-flex items-center gap-2 px-3.5 py-2 text-xs font-medium text-slate-100 bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer"
            >
              <Upload className="w-3.5 h-3.5" />
              <span>Select File</span>
            </button>
          </div>
        </div>
      </section>

      {/* Inspect Action Row */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div className="text-xs text-slate-400">
          Files are analyzed locally on the Express + Python server for schema extraction.
        </div>
        <button
          type="button"
          disabled={(!modelFile && !encoderFile) || isInspecting}
          onClick={handleRunInspection}
          className="inline-flex items-center gap-2 px-5 py-2.5 text-sm font-semibold text-slate-950 bg-cyan-400 hover:bg-cyan-300 disabled:bg-slate-800 disabled:text-slate-500 rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer disabled:cursor-not-allowed"
        >
              <FileCode2 className="w-4 h-4" />
              <span>{isInspecting ? 'Inspecting .pkl Stream...' : 'Inspect Uploaded .pkl Schema'}</span>
            </button>
          </div>
        </>
      )}

      {inspectError && (
        <div className="flex items-start gap-3 border border-rose-500/40 bg-rose-950/30 px-4 py-3 rounded-lg text-sm text-rose-200">
          <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
          <span>{inspectError}</span>
        </div>
      )}

      {/* Inspected Model Blueprint */}
      {schema ? (
        <section className="space-y-6 pt-2">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-800 pb-4">
            <div>
              <h2 className="text-xl font-semibold text-slate-100">
                Active Model Blueprint
              </h2>
              <p className="text-xs text-slate-400 mt-1 font-mono">
                {schema.modelFileName}
                {schema.encoderFileName ? ` · ${schema.encoderFileName}` : ''}
                {` · ${schema.serializationFormat}`}
              </p>
            </div>
            <div className="flex items-center gap-3">
              <button
                type="button"
                onClick={() => onNavigate('batch')}
                className="px-3.5 py-2 text-xs font-medium text-slate-200 bg-[#111827] hover:bg-slate-800 border border-slate-700 rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer"
              >
                Open Batch CSV
              </button>
              <button
                type="button"
                onClick={() => onNavigate('forecast')}
                className="px-3.5 py-2 text-xs font-semibold text-slate-950 bg-cyan-400 hover:bg-cyan-300 rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer"
              >
                Launch Forecast Console
              </button>
            </div>
          </div>

          {schema.warnings.length > 0 && (
            <div className="border border-amber-500/30 bg-amber-950/20 px-4 py-3 rounded-lg space-y-1">
              {schema.warnings.map((w, idx) => (
                <p key={idx} className="text-xs text-amber-200 flex items-start gap-2">
                  <AlertTriangle className="w-3.5 h-3.5 text-amber-400 shrink-0 mt-0.5" />
                  <span>{w}</span>
                </p>
              ))}
            </div>
          )}

          {/* Core Specs Grid */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="border border-slate-800 bg-[#111827] p-4 rounded-lg">
              <p className="text-xs text-slate-400">Estimator Class</p>
              <p className="mt-1.5 font-mono text-base font-semibold text-slate-100 truncate">
                {schema.estimatorClass}
              </p>
              <p className="mt-1 font-mono text-xs text-slate-500 truncate">
                {schema.estimatorModule}
              </p>
            </div>
            <div className="border border-slate-800 bg-[#111827] p-4 rounded-lg">
              <p className="text-xs text-slate-400">Input Signals</p>
              <p className="mt-1.5 font-mono text-2xl font-semibold text-cyan-300 tabular-nums">
                {schema.features.length}
              </p>
              <p className="mt-1 text-xs text-slate-500">Ordered feature columns</p>
            </div>
            <div className="border border-slate-800 bg-[#111827] p-4 rounded-lg">
              <p className="text-xs text-slate-400">Encoded Weather Classes</p>
              <p className="mt-1.5 font-mono text-2xl font-semibold text-slate-100 tabular-nums">
                {schema.classes.length}
              </p>
              <p className="mt-1 text-xs text-slate-500">From LabelEncoder / classes_</p>
            </div>
            <div className="border border-slate-800 bg-[#111827] p-4 rounded-lg">
              <p className="text-xs text-slate-400">Scikit-Learn Build</p>
              <p className="mt-1.5 font-mono text-base font-semibold text-slate-100 tabular-nums">
                {schema.sklearnVersion || 'Unknown'}
              </p>
              <p className="mt-1 font-mono text-xs text-slate-500 truncate">
                {schema.serializationFormat}
              </p>
            </div>
          </div>

          {/* Feature Schema + Label Encoder Mapping */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            {/* Input Features Table */}
            <div className="lg:col-span-7 border border-slate-800 bg-[#111827] rounded-lg overflow-hidden">
              <div className="flex items-center justify-between gap-4 border-b border-slate-800 px-5 py-3.5">
                <div>
                  <h3 className="text-sm font-semibold text-slate-100">
                    Input Feature Schema (X)
                  </h3>
                  <p className="text-xs text-slate-400">
                    Controls in the Forecast Console and CSV columns follow this exact order.
                  </p>
                </div>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead className="border-b border-slate-800 bg-[#0B0E17] text-slate-400 font-mono">
                    <tr>
                      <th className="px-4 py-2.5 font-medium">#</th>
                      <th className="px-4 py-2.5 font-medium">Column Key</th>
                      <th className="px-4 py-2.5 font-medium">Unit</th>
                      <th className="px-4 py-2.5 font-medium text-right">Range</th>
                      <th className="px-4 py-2.5 font-medium text-right">Action</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/80 font-mono tabular-nums">
                    {schema.features.map((feat, idx) => (
                      <tr key={feat.key} className="hover:bg-slate-800/40">
                        <td className="px-4 py-2.5 text-slate-500">{idx}</td>
                        <td className="px-4 py-2.5 text-slate-100 font-medium">
                          {feat.key}
                        </td>
                        <td className="px-4 py-2.5 text-slate-400">{feat.unit}</td>
                        <td className="px-4 py-2.5 text-right text-slate-300">
                          {feat.min} .. {feat.max}
                        </td>
                        <td className="px-4 py-2.5 text-right">
                          <button
                            type="button"
                            onClick={() => handleRemoveFeature(feat.key)}
                            disabled={schema.features.length <= 1}
                            title="Remove feature"
                            className="p-1 text-slate-500 hover:text-rose-400 disabled:opacity-30 cursor-pointer"
                          >
                            <Trash2 className="w-3.5 h-3.5" />
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Add custom feature if trained on NumPy array without feature_names_in_ */}
              <div className="border-t border-slate-800 bg-[#0B0E17] px-4 py-3 flex items-center gap-2">
                <input
                  type="text"
                  value={newFeatureName}
                  onChange={(e) => setNewFeatureName(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') handleAddFeature();
                  }}
                  placeholder="Add or customize feature column (e.g. solar_radiation)"
                  className="flex-1 bg-[#111827] border border-slate-700 rounded px-3 py-1.5 text-xs font-mono text-slate-100 placeholder:text-slate-500 focus:outline-none focus:border-cyan-400"
                />
                <button
                  type="button"
                  onClick={handleAddFeature}
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-100 bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded cursor-pointer whitespace-nowrap shrink-0"
                >
                  <Plus className="w-3.5 h-3.5" />
                  <span>Add Feature</span>
                </button>
              </div>
            </div>

            {/* Label Encoder Mapping + Pipeline/Hyperparams */}
            <div className="lg:col-span-5 space-y-6">
              <div className="border border-slate-800 bg-[#111827] rounded-lg overflow-hidden">
                <div className="border-b border-slate-800 px-5 py-3.5">
                  <h3 className="text-sm font-semibold text-slate-100">
                    Label Encoder Mapping (y)
                  </h3>
                  <p className="text-xs text-slate-400">
                    Maps integer model outputs back to human-readable weather signals.
                  </p>
                </div>

                <div className="divide-y divide-slate-800/80 font-mono text-xs tabular-nums">
                  {schema.classes.map((cls, idx) => (
                    <div
                      key={cls}
                      className="flex items-center justify-between px-5 py-2.5 hover:bg-slate-800/40"
                    >
                      <div className="flex items-center gap-4">
                        <span className="text-slate-500 w-6">[{idx}]</span>
                        <span className="text-cyan-300 font-medium">{cls}</span>
                      </div>
                      <button
                        type="button"
                        onClick={() => handleRemoveClass(cls)}
                        disabled={schema.classes.length <= 1}
                        className="p-1 text-slate-500 hover:text-rose-400 disabled:opacity-30 cursor-pointer"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  ))}
                </div>

                <div className="border-t border-slate-800 bg-[#0B0E17] px-4 py-3 flex items-center gap-2">
                  <input
                    type="text"
                    value={newClassName}
                    onChange={(e) => setNewClassName(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') handleAddClass();
                    }}
                    placeholder="Add target class label (e.g. Drizzle)"
                    className="flex-1 bg-[#111827] border border-slate-700 rounded px-3 py-1.5 text-xs font-mono text-slate-100 placeholder:text-slate-500 focus:outline-none focus:border-cyan-400"
                  />
                  <button
                    type="button"
                    onClick={handleAddClass}
                    className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-100 bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded cursor-pointer whitespace-nowrap shrink-0"
                  >
                    <Plus className="w-3.5 h-3.5" />
                    <span>Add Class</span>
                  </button>
                </div>
              </div>

              {(schema.pipelineSteps.length > 0 ||
                Object.keys(schema.hyperparameters).length > 0) && (
                <div className="border border-slate-800 bg-[#111827] rounded-lg p-5 space-y-3">
                  <h3 className="text-sm font-semibold text-slate-100">
                    Internal Structure & Hyperparameters
                  </h3>
                  {schema.pipelineSteps.length > 0 && (
                    <div className="space-y-1.5">
                      <p className="text-xs text-slate-400">Pipeline Stages</p>
                      <div className="font-mono text-xs text-slate-200 space-y-1">
                        {schema.pipelineSteps.map((st, idx) => (
                          <div key={st.name} className="flex items-center gap-2">
                            <span className="text-slate-500">{idx + 1}.</span>
                            <span>{st.name}</span>
                            <span className="text-slate-600">→</span>
                            <span className="text-cyan-300">{st.class}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                  {Object.keys(schema.hyperparameters).length > 0 && (
                    <div className="grid grid-cols-2 gap-x-4 gap-y-1.5 pt-2 border-t border-slate-800 font-mono text-xs tabular-nums">
                      {Object.entries(schema.hyperparameters).map(([k, v]) => (
                        <div key={k} className="flex items-center justify-between gap-2">
                          <span className="text-slate-400 truncate">{k}</span>
                          <span className="text-slate-100">{String(v)}</span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </section>
      ) : (
        <section className="border border-slate-800 bg-[#111827] rounded-lg p-8 text-center space-y-4">
          <h2 className="text-lg font-semibold text-slate-100">
            No Model Schema Loaded Yet
          </h2>
          <p className="mx-auto max-w-xl text-sm text-slate-400">
            Upload your <span className="font-mono text-slate-200">weather_signal_model.pkl</span> and <span className="font-mono text-slate-200">label_encoder.pkl</span> above to inspect their structure, or load the sample meteorological classifier to explore the Forecast Console and CSV Batch Analyzer immediately.
          </p>
          <div>
            <button
              type="button"
              onClick={onLoadSample}
              className="px-5 py-2.5 text-sm font-semibold text-slate-950 bg-cyan-400 hover:bg-cyan-300 rounded-md transition-colors cursor-pointer"
            >
              Load Sample Weather Model
            </button>
          </div>
        </section>
      )}
    </div>
  );
};
