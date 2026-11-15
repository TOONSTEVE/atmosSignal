import React, { useMemo, useState } from 'react';
import {
  AlertTriangle,
  Cloud,
  CloudFog,
  CloudLightning,
  CloudRain,
  RotateCcw,
  Sun,
} from 'lucide-react';
import type { ForecastResult, ModelSchema } from '../types/weather';
import { WEATHER_PRESETS } from '../data/sampleModel';

interface ForecastConsoleProps {
  schema: ModelSchema | null;
  inputs: Record<string, number>;
  forecast: ForecastResult | null;
  isPredicting: boolean;
  sweepFeature: string;
  onChangeInput: (key: string, value: number) => void;
  onApplyPreset: (values: Record<string, number>) => void;
  onResetNominal: () => void;
  onChangeSweepFeature: (key: string) => void;
  onLoadSample: () => void;
}

const CLASS_COLORS = [
  '#06B6D4', // cyan
  '#F59E0B', // amber
  '#10B981', // emerald
  '#818CF8', // indigo
  '#F43F5E', // rose
  '#38BDF8', // sky
];

function getWeatherIcon(label: string) {
  const lower = label.toLowerCase();
  if (lower.includes('storm') || lower.includes('thunder')) {
    return <CloudLightning className="w-7 h-7 text-amber-400" />;
  }
  if (lower.includes('rain') || lower.includes('shower') || lower.includes('wet')) {
    return <CloudRain className="w-7 h-7 text-cyan-400" />;
  }
  if (lower.includes('fog') || lower.includes('mist') || lower.includes('haze')) {
    return <CloudFog className="w-7 h-7 text-slate-300" />;
  }
  if (lower.includes('overcast') || lower.includes('cloud')) {
    return <Cloud className="w-7 h-7 text-slate-300" />;
  }
  return <Sun className="w-7 h-7 text-emerald-400" />;
}

export const ForecastConsole: React.FC<ForecastConsoleProps> = ({
  schema,
  inputs,
  forecast,
  isPredicting,
  sweepFeature,
  onChangeInput,
  onApplyPreset,
  onResetNominal,
  onChangeSweepFeature,
  onLoadSample,
}) => {
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);

  const radarData = useMemo(() => {
    if (!schema) return [];
    const axes = schema.features.slice(0, 8);
    const total = axes.length;
    const cx = 140;
    const cy = 135;
    const radius = 88;

    return axes.map((f, idx) => {
      const angle = -Math.PI / 2 + (idx * 2 * Math.PI) / total;
      const span = f.max - f.min || 1;
      const currentVal = inputs[f.key] ?? f.defaultValue;
      const normCurrent = Math.max(0.08, Math.min(1, (currentVal - f.min) / span));
      const normNominal = Math.max(0.08, Math.min(1, (f.nominalCenter - f.min) / span));

      return {
        key: f.key,
        label: f.key,
        unit: f.unit,
        value: currentVal,
        xOuter: cx + radius * Math.cos(angle),
        yOuter: cy + radius * Math.sin(angle),
        xLabel: cx + (radius + 22) * Math.cos(angle),
        yLabel: cy + (radius + 22) * Math.sin(angle),
        xCurrent: cx + radius * normCurrent * Math.cos(angle),
        yCurrent: cy + radius * normCurrent * Math.sin(angle),
        xNominal: cx + radius * normNominal * Math.cos(angle),
        yNominal: cy + radius * normNominal * Math.sin(angle),
      };
    });
  }, [schema, inputs]);

  if (!schema) {
    return (
      <div className="mx-auto max-w-[1440px] px-6 py-16">
        <div className="border border-slate-800 bg-[#111827] rounded-lg p-10 text-center space-y-4">
          <h2 className="text-xl font-semibold text-slate-100">
            Forecast Console Requires an Active Model Schema
          </h2>
          <p className="mx-auto max-w-lg text-sm text-slate-400">
            Inspect your <span className="font-mono text-slate-200">weather_signal_model.pkl</span> in the Inspector tab or load the sample meteorological classifier to unlock live signal sliders and probability telemetry.
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

  const currentPolygon = radarData.map((d) => `${d.xCurrent.toFixed(1)},${d.yCurrent.toFixed(1)}`).join(' ');
  const nominalPolygon = radarData.map((d) => `${d.xNominal.toFixed(1)},${d.yNominal.toFixed(1)}`).join(' ');

  const series = forecast?.sensitivitySeries || [];
  const activeSweepSpec = schema.features.find((f) => f.key === sweepFeature) || schema.features[0];
  const activeClasses = (forecast?.probabilities || []).slice(0, 5).map((p) => p.label);

  // SVG chart dimensions for sensitivity plot
  const chartW = 460;
  const chartH = 220;
  const padL = 42;
  const padR = 16;
  const padT = 18;
  const padB = 34;
  const innerW = chartW - padL - padR;
  const innerH = chartH - padT - padB;

  const activeHoverPoint =
    hoverIndex !== null && series[hoverIndex] ? series[hoverIndex] : null;

  return (
    <div className="mx-auto max-w-[1440px] px-6 py-6">
      <div className="flex flex-col lg:flex-row gap-6 items-start">
        {/* Left Control & Parameter Column */}
        <aside className="w-full lg:w-[360px] shrink-0 border border-slate-800 bg-[#111827] rounded-lg overflow-hidden">
          <div className="flex items-center justify-between border-b border-slate-800 px-5 py-3.5">
            <div>
              <h2 className="text-sm font-semibold text-slate-100">
                Signal Controls
              </h2>
              <p className="text-xs font-mono text-slate-400">
                {schema.features.length} input features
              </p>
            </div>
            <button
              type="button"
              onClick={onResetNominal}
              className="inline-flex items-center gap-1.5 px-2.5 py-1.5 text-xs font-medium text-slate-300 hover:text-slate-100 bg-slate-800/70 hover:bg-slate-800 border border-slate-700 rounded cursor-pointer"
            >
              <RotateCcw className="w-3 h-3" />
              <span>Reset</span>
            </button>
          </div>

          {/* Atmospheric Presets */}
          <div className="border-b border-slate-800 px-5 py-3.5 space-y-2">
            <p className="text-xs font-medium text-slate-400">
              Atmospheric Presets
            </p>
            <div className="grid grid-cols-1 gap-1.5">
              {WEATHER_PRESETS.map((preset) => (
                <button
                  key={preset.id}
                  type="button"
                  onClick={() => onApplyPreset(preset.values)}
                  className="text-left px-3 py-2 rounded border border-slate-800 bg-[#0B0E17] hover:border-cyan-500/50 hover:bg-slate-900 transition-colors cursor-pointer"
                >
                  <div className="text-xs font-medium text-slate-100">
                    {preset.name}
                  </div>
                  <div className="text-[11px] text-slate-500 truncate">
                    {preset.summary}
                  </div>
                </button>
              ))}
            </div>
          </div>

          {/* Dynamic Feature Sliders */}
          <div className="divide-y divide-slate-800/80 max-h-[540px] overflow-y-auto">
            {schema.features.map((feat) => {
              const val = inputs[feat.key] ?? feat.defaultValue;
              return (
                <div key={feat.key} className="px-5 py-3.5 space-y-2">
                  <div className="flex items-center justify-between gap-2">
                    <label
                      htmlFor={`slider-${feat.key}`}
                      className="text-xs font-medium text-slate-200 truncate"
                      title={feat.key}
                    >
                      {feat.label}
                    </label>
                    <div className="flex items-center gap-1.5 shrink-0">
                      <input
                        type="number"
                        value={val}
                        min={feat.min}
                        max={feat.max}
                        step={feat.step}
                        onChange={(e) => {
                          const parsed = parseFloat(e.target.value);
                          if (!Number.isNaN(parsed)) {
                            onChangeInput(feat.key, parsed);
                          }
                        }}
                        className="w-20 bg-[#0B0E17] border border-slate-700 rounded px-2 py-1 text-right font-mono text-xs text-cyan-300 tabular-nums focus:outline-none focus:border-cyan-400"
                      />
                      <span className="font-mono text-[11px] text-slate-500 w-9">
                        {feat.unit}
                      </span>
                    </div>
                  </div>
                  <input
                    id={`slider-${feat.key}`}
                    type="range"
                    min={feat.min}
                    max={feat.max}
                    step={feat.step}
                    value={Math.min(feat.max, Math.max(feat.min, val))}
                    onChange={(e) => onChangeInput(feat.key, parseFloat(e.target.value))}
                    className="w-full telemetry-slider"
                  />
                  <div className="flex justify-between font-mono text-[10px] text-slate-500 tabular-nums">
                    <span>
                      {feat.min} {feat.unit}
                    </span>
                    <span className="text-slate-600">{feat.key}</span>
                    <span>
                      {feat.max} {feat.unit}
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        </aside>

        {/* Main Viewport Stage */}
        <main className="flex-1 min-w-0 space-y-6 w-full">
          {/* Primary Forecast Output Banner */}
          <section className="border border-slate-800 bg-[#111827] rounded-lg p-6">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-6">
              <div className="flex items-start gap-4">
                <div className="p-3 rounded-lg border border-slate-800 bg-[#0B0E17] shrink-0">
                  {getWeatherIcon(forecast?.predictedLabel || 'Clear')}
                </div>
                <div className="space-y-1">
                  <div className="flex flex-wrap items-center gap-2 text-xs font-mono text-slate-400">
                    <span>Predicted Weather Signal</span>
                    <span aria-hidden="true">·</span>
                    <span>
                      {forecast?.mode === 'live-sklearn'
                        ? 'Live Scikit-Learn Inference'
                        : forecast?.mode === 'native-json-model'
                          ? 'Exact Trained Model (JSON)'
                          : 'Calibrated Signal Engine'}
                    </span>
                    {isPredicting && (
                      <>
                        <span aria-hidden="true">·</span>
                        <span className="text-cyan-400">Updating</span>
                      </>
                    )}
                  </div>
                  <h2 className="text-3xl sm:text-4xl font-semibold tracking-tight text-slate-50">
                    {forecast?.predictedLabel || 'Evaluating...'}
                  </h2>
                  <p className="text-sm text-slate-400 max-w-2xl">
                    {forecast?.advisoryNote ||
                      'Adjust any surface telemetry parameter on the left to evaluate class probabilities.'}
                  </p>
                </div>
              </div>

              <div className="border-t md:border-t-0 md:border-l border-slate-800 pt-4 md:pt-0 md:pl-6 shrink-0">
                <p className="text-xs font-mono uppercase tracking-wider text-slate-400">
                  Model Confidence
                </p>
                <p className="mt-1 font-mono text-3xl font-bold text-cyan-300 tabular-nums">
                  {forecast ? `${(forecast.confidence * 100).toFixed(1)}%` : '--'}
                </p>
                <p className="mt-1 font-mono text-xs text-slate-500">
                  {schema.estimatorClass}
                </p>
              </div>
            </div>

            {forecast?.mode !== 'live-sklearn' && schema.source === 'inspected-pkl' && (
              <div className="mt-5 pt-4 border-t border-slate-800 flex items-start gap-2.5 text-xs text-amber-200">
                <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
                <span>
                  Your `.pkl` schema is loaded, and this container lacks `scikit-learn` for native binary unpickling. Use the Python Bridge tab to run `predict_api.py` locally for exact tree/weight execution.
                </span>
              </div>
            )}
          </section>

          {/* Class Probability Distribution */}
          <section className="border border-slate-800 bg-[#111827] rounded-lg p-6 space-y-4">
            <div className="flex items-center justify-between gap-4">
              <div>
                <h3 className="text-sm font-semibold text-slate-100">
                  Encoded Class Probability Distribution
                </h3>
                <p className="text-xs text-slate-400">
                  Softmax / `predict_proba` likelihood across `label_encoder.pkl` classes
                </p>
              </div>
              <span className="font-mono text-xs text-slate-500 tabular-nums">
                {schema.classes.length} classes
              </span>
            </div>

            <div className="space-y-3">
              {(forecast?.probabilities || []).map((item, idx) => {
                const pct = Math.max(0, Math.min(100, item.probability * 100));
                const isTop = item.label === forecast?.predictedLabel;
                const barColor = CLASS_COLORS[idx % CLASS_COLORS.length];
                return (
                  <div key={item.label} className="space-y-1.5">
                    <div className="flex items-center justify-between text-xs font-mono tabular-nums">
                      <span
                        className={
                          isTop ? 'font-semibold text-slate-100' : 'text-slate-400'
                        }
                      >
                        {item.label}
                      </span>
                      <span
                        className={
                          isTop ? 'font-semibold text-cyan-300' : 'text-slate-300'
                        }
                      >
                        {pct.toFixed(1)}%
                      </span>
                    </div>
                    <div className="h-2 w-full rounded bg-[#0B0E17] border border-slate-800 overflow-hidden">
                      <div
                        className="h-full transition-all duration-150"
                        style={{
                          width: `${pct}%`,
                          backgroundColor: isTop ? '#06B6D4' : barColor,
                          opacity: isTop ? 1 : 0.6,
                        }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </section>

          {/* Scientific Plots Grid: Radar Calibration + Sensitivity Sweep */}
          <section className="grid grid-cols-1 xl:grid-cols-12 gap-6">
            {/* Radar Calibration Polygon */}
            <div className="xl:col-span-5 border border-slate-800 bg-[#111827] rounded-lg p-5 space-y-3">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-sm font-semibold text-slate-100">
                    Signal Calibration Radar
                  </h3>
                  <p className="text-xs text-slate-400">
                    Current vector vs. nominal baseline
                  </p>
                </div>
              </div>

              <div className="flex justify-center">
                <svg
                  viewBox="0 0 280 270"
                  className="w-full max-w-[290px] h-auto"
                  role="img"
                  aria-label="Signal calibration radar chart"
                >
                  {[0.25, 0.5, 0.75, 1].map((ring) => {
                    const pts = radarData
                      .map((d) => {
                        const dx = d.xOuter - 140;
                        const dy = d.yOuter - 135;
                        return `${(140 + dx * ring).toFixed(1)},${(135 + dy * ring).toFixed(1)}`;
                      })
                      .join(' ');
                    return (
                      <polygon
                        key={ring}
                        points={pts}
                        fill="none"
                        stroke="#1E293B"
                        strokeWidth="1"
                      />
                    );
                  })}

                  {radarData.map((d) => (
                    <g key={d.key}>
                      <line
                        x1="140"
                        y1="135"
                        x2={d.xOuter}
                        y2={d.yOuter}
                        stroke="#1E293B"
                        strokeWidth="1"
                      />
                      <text
                        x={d.xLabel}
                        y={d.yLabel}
                        textAnchor="middle"
                        dominantBaseline="middle"
                        className="fill-slate-400 font-mono text-[8.5px]"
                      >
                        {d.label.length > 11 ? `${d.label.slice(0, 10)}…` : d.label}
                      </text>
                    </g>
                  ))}

                  {nominalPolygon && (
                    <polygon
                      points={nominalPolygon}
                      fill="rgba(148, 163, 184, 0.08)"
                      stroke="#64748B"
                      strokeDasharray="3 3"
                      strokeWidth="1.25"
                    />
                  )}

                  {currentPolygon && (
                    <polygon
                      points={currentPolygon}
                      fill="rgba(6, 182, 212, 0.2)"
                      stroke="#06B6D4"
                      strokeWidth="2"
                    />
                  )}
                </svg>
              </div>

              <div className="flex items-center justify-center gap-6 text-xs font-mono text-slate-400 pt-1">
                <span className="flex items-center gap-2">
                  <span className="w-3 h-0.5 bg-cyan-400 inline-block" />
                  Current Signal
                </span>
                <span className="flex items-center gap-2">
                  <span className="w-3 h-0.5 border-t border-dashed border-slate-400 inline-block" />
                  Nominal Baseline
                </span>
              </div>
            </div>

            {/* Sensitivity Curve */}
            <div className="xl:col-span-7 border border-slate-800 bg-[#111827] rounded-lg p-5 space-y-4">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                <div>
                  <h3 className="text-sm font-semibold text-slate-100">
                    Parameter Sensitivity Sweep
                  </h3>
                  <p className="text-xs text-slate-400">
                    Class probability response across {activeSweepSpec?.label} range
                  </p>
                </div>

                <select
                  aria-label="Select sweep parameter"
                  value={sweepFeature}
                  onChange={(e) => onChangeSweepFeature(e.target.value)}
                  className="bg-[#0B0E17] border border-slate-700 rounded px-3 py-1.5 text-xs font-mono text-slate-200 focus:outline-none focus:border-cyan-400 cursor-pointer"
                >
                  {schema.features.map((f) => (
                    <option key={f.key} value={f.key}>
                      Sweep: {f.key}
                    </option>
                  ))}
                </select>
              </div>

              <div className="relative">
                <svg
                  viewBox={`0 0 ${chartW} ${chartH}`}
                  className="w-full h-auto cursor-crosshair"
                  onMouseLeave={() => setHoverIndex(null)}
                  onMouseMove={(e) => {
                    if (series.length < 2) return;
                    const rect = e.currentTarget.getBoundingClientRect();
                    const relX = ((e.clientX - rect.left) / rect.width) * chartW;
                    const clamped = Math.max(padL, Math.min(chartW - padR, relX));
                    const idx = Math.round(((clamped - padL) / innerW) * (series.length - 1));
                    setHoverIndex(Math.max(0, Math.min(series.length - 1, idx)));
                  }}
                  role="img"
                  aria-label="Class probability sensitivity sweep curve"
                >
                  {/* Y Gridlines */}
                  {[0, 0.25, 0.5, 0.75, 1].map((tick) => {
                    const y = padT + innerH * (1 - tick);
                    return (
                      <g key={tick}>
                        <line
                          x1={padL}
                          y1={y}
                          x2={chartW - padR}
                          y2={y}
                          stroke="#1E293B"
                          strokeWidth="1"
                        />
                        <text
                          x={padL - 8}
                          y={y}
                          textAnchor="end"
                          dominantBaseline="middle"
                          className="fill-slate-500 font-mono text-[9px]"
                        >
                          {Math.round(tick * 100)}%
                        </text>
                      </g>
                    );
                  })}

                  {/* X Axis Min / Mid / Max */}
                  {series.length > 1 &&
                    [0, Math.floor(series.length / 2), series.length - 1].map((idx) => {
                      const x = padL + (idx / (series.length - 1)) * innerW;
                      return (
                        <text
                          key={idx}
                          x={x}
                          y={chartH - 10}
                          textAnchor="middle"
                          className="fill-slate-500 font-mono text-[9px]"
                        >
                          {series[idx].xValue}
                          {activeSweepSpec?.unit ? ` ${activeSweepSpec.unit}` : ''}
                        </text>
                      );
                    })}

                  {/* Class Lines */}
                  {series.length > 1 &&
                    activeClasses.map((cls, cIdx) => {
                      const pts = series
                        .map((pt, i) => {
                          const x = padL + (i / (series.length - 1)) * innerW;
                          const pVal = pt.probabilities[cls] ?? 0;
                          const y = padT + innerH * (1 - pVal);
                          return `${x.toFixed(1)},${y.toFixed(1)}`;
                        })
                        .join(' ');
                      return (
                        <polyline
                          key={cls}
                          fill="none"
                          stroke={CLASS_COLORS[cIdx % CLASS_COLORS.length]}
                          strokeWidth={cls === forecast?.predictedLabel ? 2.4 : 1.5}
                          strokeOpacity={cls === forecast?.predictedLabel ? 1 : 0.7}
                          points={pts}
                        />
                      );
                    })}

                  {/* Current Parameter Marker */}
                  {activeSweepSpec && series.length > 1 && (() => {
                    const currentVal = inputs[activeSweepSpec.key] ?? activeSweepSpec.defaultValue;
                    const ratio = Math.max(
                      0,
                      Math.min(
                        1,
                        (currentVal - activeSweepSpec.min) /
                          (activeSweepSpec.max - activeSweepSpec.min || 1)
                      )
                    );
                    const markerX = padL + ratio * innerW;
                    return (
                      <line
                        x1={markerX}
                        y1={padT}
                        x2={markerX}
                        y2={padT + innerH}
                        stroke="#94A3B8"
                        strokeDasharray="3 3"
                        strokeWidth="1"
                      />
                    );
                  })()}

                  {/* Hover Crosshair */}
                  {hoverIndex !== null && series.length > 1 && (
                    <line
                      x1={padL + (hoverIndex / (series.length - 1)) * innerW}
                      y1={padT}
                      x2={padL + (hoverIndex / (series.length - 1)) * innerW}
                      y2={padT + innerH}
                      stroke="#22D3EE"
                      strokeWidth="1"
                    />
                  )}
                </svg>

                {activeHoverPoint && (
                  <div className="mt-2 border border-slate-700 bg-[#0B0E17] px-3 py-2 rounded font-mono text-[11px] flex flex-wrap items-center gap-x-4 gap-y-1 tabular-nums">
                    <span className="text-cyan-300 font-semibold">
                      {activeSweepSpec?.key}: {activeHoverPoint.xValue} {activeSweepSpec?.unit}
                    </span>
                    {activeClasses.map((cls, cIdx) => (
                      <span key={cls} className="flex items-center gap-1.5 text-slate-300">
                        <span
                          className="w-2 h-2 rounded-xs inline-block"
                          style={{ backgroundColor: CLASS_COLORS[cIdx % CLASS_COLORS.length] }}
                        />
                        {cls}: {((activeHoverPoint.probabilities[cls] ?? 0) * 100).toFixed(1)}%
                      </span>
                    ))}
                  </div>
                )}
              </div>

              <div className="flex flex-wrap items-center gap-x-5 gap-y-1.5 text-xs font-mono text-slate-400">
                {activeClasses.map((cls, cIdx) => (
                  <span key={cls} className="flex items-center gap-1.5">
                    <span
                      className="w-2.5 h-0.5 inline-block"
                      style={{ backgroundColor: CLASS_COLORS[cIdx % CLASS_COLORS.length] }}
                    />
                    {cls}
                  </span>
                ))}
              </div>
            </div>
          </section>
        </main>
      </div>
    </div>
  );
};
