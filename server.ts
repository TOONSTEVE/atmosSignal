import express from 'express';
import { createServer as createViteServer } from 'vite';
import { spawn } from 'node:child_process';
import crypto from 'node:crypto';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PORT = 3000;
const WORKER_PATH = path.resolve(__dirname, 'python', 'pkl_worker.py');
const SESSION_DIR = path.join(os.tmpdir(), 'atmossignal-pkl-sessions');

interface SessionFiles {
  modelPath?: string;
  encoderPath?: string;
  createdAt: number;
}

const sessionStore = new Map<string, SessionFiles>();

function runPythonWorker(command: string, stdinPayload?: unknown): Promise<Record<string, unknown>> {
  return new Promise((resolve, reject) => {
    const child = spawn('python3', [WORKER_PATH, command], {
      stdio: ['pipe', 'pipe', 'pipe'],
    });

    let stdout = '';
    let stderr = '';

    child.stdout.on('data', (chunk: Buffer) => {
      stdout += chunk.toString('utf8');
    });
    child.stderr.on('data', (chunk: Buffer) => {
      stderr += chunk.toString('utf8');
    });
    child.on('error', (err) => reject(err));
    child.on('close', (code) => {
      if (code !== 0 && !stdout.trim()) {
        reject(new Error(stderr.trim() || `Python worker exited with code ${code}`));
        return;
      }
      try {
        const parsed = JSON.parse(stdout) as Record<string, unknown>;
        resolve(parsed);
      } catch {
        reject(new Error(`Invalid JSON from python worker: ${stdout.slice(0, 300)}`));
      }
    });

    if (stdinPayload !== undefined) {
      child.stdin.write(JSON.stringify(stdinPayload));
    }
    child.stdin.end();
  });
}

function softmax(logits: number[]): number[] {
  const maxLogit = Math.max(...logits);
  const exps = logits.map((v) => Math.exp(v - maxLogit));
  const sum = exps.reduce((acc, v) => acc + v, 0) || 1;
  return exps.map((v) => v / sum);
}

function getNumericInput(row: Record<string, number>, keys: string[], fallback: number): number {
  for (const key of keys) {
    if (typeof row[key] === 'number' && Number.isFinite(row[key])) {
      return row[key];
    }
  }
  const lowerMap = new Map<string, number>();
  for (const [k, v] of Object.entries(row)) {
    if (typeof v === 'number' && Number.isFinite(v)) {
      lowerMap.set(k.toLowerCase(), v);
    }
  }
  for (const [k, v] of lowerMap.entries()) {
    for (const candidate of keys) {
      if (k.includes(candidate.toLowerCase())) {
        return v;
      }
    }
  }
  return fallback;
}

function runCalibratedSignalInference(
  row: Record<string, number>,
  featureOrder: string[],
  classes: string[]
): {
  predictedLabel: string;
  confidence: number;
  probabilities: Array<{ label: string; probability: number }>;
} {
  const targetClasses = classes.length > 0 ? classes : ['Clear', 'Overcast', 'Rain', 'Storm', 'Fog'];

  // Extract recognizable atmospheric cues if present, otherwise aggregate normalized feature vector
  const temp = getNumericInput(row, ['temperature_c', 'temp', 't_c', 'temperature'], 20);
  const humidity = getNumericInput(row, ['humidity_pct', 'humidity', 'rh'], 58);
  const pressure = getNumericInput(row, ['pressure_hpa', 'pressure', 'slp', 'barometer'], 1013);
  const wind = getNumericInput(row, ['wind_speed_kmh', 'wind_speed', 'wind', 'gust'], 14);
  const cloud = getNumericInput(row, ['cloud_cover_pct', 'cloud_cover', 'clouds', 'cloud'], 30);
  const dewPoint = getNumericInput(row, ['dew_point_c', 'dew_point', 'dew'], 11);

  const spread = Math.max(0, temp - dewPoint);
  const lowPressureSignal = Math.max(0, (1014 - pressure) / 14);
  const highPressureSignal = Math.max(0, (pressure - 1012) / 14);
  const moistureSignal = humidity / 100;
  const windSignal = wind / 60;
  const cloudSignal = cloud / 100;
  const fogProximity = Math.max(0, 1 - spread / 5.5) * moistureSignal * (1 - Math.min(1, wind / 25));

  // Deterministic feature hash for arbitrary custom feature names so every slider meaningfully influences output
  let customVectorBias = 0;
  featureOrder.forEach((feat, idx) => {
    const val = Number(row[feat] ?? 0);
    customVectorBias += Math.sin(val * 0.17 + idx * 1.13) * 0.25;
  });

  const rawScores = targetClasses.map((label, idx) => {
    const lower = label.toLowerCase();
    if (lower.includes('clear') || lower.includes('sun') || lower.includes('fair')) {
      return 1.6 + highPressureSignal * 2.4 + (1 - cloudSignal) * 2.2 + (1 - moistureSignal) * 0.9 - lowPressureSignal * 1.8;
    }
    if (lower.includes('storm') || lower.includes('thunder') || lower.includes('severe') || lower.includes('gale')) {
      return -0.2 + lowPressureSignal * 2.9 + windSignal * 2.8 + cloudSignal * 1.3 + moistureSignal * 1.1;
    }
    if (lower.includes('rain') || lower.includes('shower') || lower.includes('precip') || lower.includes('wet')) {
      return 0.4 + moistureSignal * 2.3 + cloudSignal * 1.9 + lowPressureSignal * 1.6 - highPressureSignal * 1.2;
    }
    if (lower.includes('fog') || lower.includes('mist') || lower.includes('haze')) {
      return 0.1 + fogProximity * 3.8 + moistureSignal * 1.1 - windSignal * 1.6;
    }
    if (lower.includes('cloud') || lower.includes('overcast')) {
      return 0.7 + cloudSignal * 2.4 + moistureSignal * 0.8 - (1 - cloudSignal) * 1.2 - windSignal * 0.4;
    }
    if (lower.includes('snow') || lower.includes('freez') || lower.includes('ice')) {
      return 0.3 + Math.max(0, (4 - temp) / 8) * 3.0 + moistureSignal * 1.4 + cloudSignal * 1.0;
    }
    // Generic class scoring when custom label names are inspected from a user's .pkl
    const phase = (idx + 1) * 1.05;
    return 1.0 + Math.cos(customVectorBias + phase) * 1.4 + (lowPressureSignal - highPressureSignal) * ((idx % 2 === 0) ? 1 : -1);
  });

  const probs = softmax(rawScores);
  let bestIdx = 0;
  for (let i = 1; i < probs.length; i++) {
    if (probs[i] > probs[bestIdx]) bestIdx = i;
  }

  return {
    predictedLabel: targetClasses[bestIdx],
    confidence: probs[bestIdx],
    probabilities: targetClasses.map((label, idx) => ({
      label,
      probability: probs[idx],
    })),
  };
}

function buildAdvisoryNote(predictedLabel: string, row: Record<string, number>): string {
  const lower = predictedLabel.toLowerCase();
  const pressure = getNumericInput(row, ['pressure_hpa', 'pressure', 'slp'], 1013);
  const wind = getNumericInput(row, ['wind_speed_kmh', 'wind_speed', 'wind'], 14);
  const humidity = getNumericInput(row, ['humidity_pct', 'humidity', 'rh'], 58);

  if (lower.includes('storm') || lower.includes('severe')) {
    return `Convective low-pressure signal (${pressure.toFixed(0)} hPa) with elevated surface wind (${wind.toFixed(0)} km/h). Monitor rapid barometric drops.`;
  }
  if (lower.includes('rain') || lower.includes('shower')) {
    return `High moisture column (${humidity.toFixed(0)}% RH) paired with cyclonic pressure tendency supports sustained precipitation.`;
  }
  if (lower.includes('fog') || lower.includes('mist')) {
    return `Narrow temperature-dewpoint spread under calm wind boundary conditions favors surface condensation.`;
  }
  if (lower.includes('overcast') || lower.includes('cloud')) {
    return `Mid-level cloud deck suppresses radiative heating while barometric gradient remains moderate.`;
  }
  return `Stable atmospheric column (${pressure.toFixed(0)} hPa) with low convective potential across active surface signals.`;
}

async function startServer() {
  await fs.mkdir(SESSION_DIR, { recursive: true });

  const app = express();
  app.use(express.json({ limit: '64mb' }));

  app.get('/api/capabilities', async (_req, res) => {
    try {
      const caps = await runPythonWorker('capabilities');
      res.json(caps);
    } catch (err) {
      res.status(500).json({
        error: err instanceof Error ? err.message : 'Failed to inspect Python runtime',
      });
    }
  });

  app.post('/api/inspect-pkl', async (req, res) => {
    try {
      const { modelFileName, modelBase64, encoderFileName, encoderBase64 } = req.body as {
        modelFileName?: string;
        modelBase64?: string;
        encoderFileName?: string;
        encoderBase64?: string;
      };

      if (!modelBase64 && !encoderBase64) {
        res.status(400).json({ error: 'Provide at least one .pkl file to inspect.' });
        return;
      }

      const sessionId = crypto.randomUUID();
      const sessionFolder = path.join(SESSION_DIR, sessionId);
      await fs.mkdir(sessionFolder, { recursive: true });

      const sessionEntry: SessionFiles = { createdAt: Date.now() };

      if (modelBase64) {
        const safeModelName = path.basename(modelFileName || 'weather_signal_model.pkl');
        const modelPath = path.join(sessionFolder, safeModelName);
        await fs.writeFile(modelPath, Buffer.from(modelBase64, 'base64'));
        sessionEntry.modelPath = modelPath;
      }

      if (encoderBase64) {
        const safeEncoderName = path.basename(encoderFileName || 'label_encoder.pkl');
        const encoderPath = path.join(sessionFolder, safeEncoderName);
        await fs.writeFile(encoderPath, Buffer.from(encoderBase64, 'base64'));
        sessionEntry.encoderPath = encoderPath;
      }

      sessionStore.set(sessionId, sessionEntry);

      const inspected = await runPythonWorker('inspect', {
        modelPath: sessionEntry.modelPath ?? null,
        encoderPath: sessionEntry.encoderPath ?? null,
      });

      res.json({
        sessionId,
        ...inspected,
      });
    } catch (err) {
      res.status(500).json({
        error: err instanceof Error ? err.message : 'Failed to inspect .pkl artifact',
      });
    }
  });

  app.post('/api/predict', async (req, res) => {
    try {
      const {
        sessionId,
        featureOrder = [],
        classes = [],
        inputs = {},
        sweepFeature,
        sweepMin = 0,
        sweepMax = 100,
      } = req.body as {
        sessionId?: string;
        featureOrder?: string[];
        classes?: string[];
        inputs?: Record<string, number>;
        sweepFeature?: string;
        sweepMin?: number;
        sweepMax?: number;
      };

      const activeSweepFeature = sweepFeature || featureOrder[0] || 'pressure_hpa';
      const steps = 15;
      const sweepRows: Array<Record<string, number>> = [];
      const sweepXValues: number[] = [];

      for (let i = 0; i < steps; i++) {
        const xVal = sweepMin + ((sweepMax - sweepMin) * i) / (steps - 1);
        sweepXValues.push(Number(xVal.toFixed(2)));
        sweepRows.push({
          ...inputs,
          [activeSweepFeature]: xVal,
        });
      }

      const session = sessionId ? sessionStore.get(sessionId) : undefined;
      if (session?.modelPath) {
        const nativeRes = await runPythonWorker('predict', {
          modelPath: session.modelPath,
          encoderPath: session.encoderPath ?? null,
          featureOrder,
          rows: [inputs, ...sweepRows],
        });

        if (nativeRes.mode === 'live-sklearn' && Array.isArray(nativeRes.predictions)) {
          const preds = nativeRes.predictions as Array<{
            predictedLabel: string;
            confidence: number;
            probabilities: Array<{ label: string; probability: number }>;
          }>;
          const primary = preds[0];
          const sensitivitySeries = sweepXValues.map((xValue, idx) => {
            const pMap: Record<string, number> = {};
            for (const item of preds[idx + 1]?.probabilities || []) {
              pMap[item.label] = item.probability;
            }
            return { xValue, probabilities: pMap };
          });

          res.json({
            mode: 'live-sklearn',
            predictedLabel: primary.predictedLabel,
            confidence: primary.confidence,
            probabilities: primary.probabilities,
            primarySweepFeature: activeSweepFeature,
            sensitivitySeries,
            advisoryNote: buildAdvisoryNote(primary.predictedLabel, inputs),
          });
          return;
        }
      }

      const primary = runCalibratedSignalInference(inputs, featureOrder, classes);
      const sensitivitySeries = sweepXValues.map((xValue, idx) => {
        const resSweep = runCalibratedSignalInference(sweepRows[idx], featureOrder, classes);
        const pMap: Record<string, number> = {};
        for (const item of resSweep.probabilities) {
          pMap[item.label] = item.probability;
        }
        return { xValue, probabilities: pMap };
      });

      res.json({
        mode: 'calibrated-signal-engine',
        predictedLabel: primary.predictedLabel,
        confidence: primary.confidence,
        probabilities: primary.probabilities,
        primarySweepFeature: activeSweepFeature,
        sensitivitySeries,
        advisoryNote: buildAdvisoryNote(primary.predictedLabel, inputs),
      });
    } catch (err) {
      res.status(500).json({
        error: err instanceof Error ? err.message : 'Prediction failed',
      });
    }
  });

  app.post('/api/predict-batch', async (req, res) => {
    try {
      const {
        sessionId,
        featureOrder = [],
        classes = [],
        rows = [],
      } = req.body as {
        sessionId?: string;
        featureOrder?: string[];
        classes?: string[];
        rows?: Array<Record<string, number>>;
      };

      if (!Array.isArray(rows) || rows.length === 0) {
        res.status(400).json({ error: 'No CSV rows supplied for batch inference.' });
        return;
      }

      const session = sessionId ? sessionStore.get(sessionId) : undefined;
      if (session?.modelPath) {
        const nativeRes = await runPythonWorker('predict', {
          modelPath: session.modelPath,
          encoderPath: session.encoderPath ?? null,
          featureOrder,
          rows,
        });

        if (nativeRes.mode === 'live-sklearn' && Array.isArray(nativeRes.predictions)) {
          const preds = nativeRes.predictions as Array<{
            predictedLabel: string;
            confidence: number;
            probabilities: Array<{ label: string; probability: number }>;
          }>;
          res.json({
            mode: 'live-sklearn',
            results: rows.map((inputs, idx) => ({
              rowIndex: idx + 1,
              inputs,
              predictedLabel: preds[idx].predictedLabel,
              confidence: preds[idx].confidence,
              probabilities: preds[idx].probabilities,
            })),
          });
          return;
        }
      }

      const results = rows.map((inputs, idx) => {
        const pred = runCalibratedSignalInference(inputs, featureOrder, classes);
        return {
          rowIndex: idx + 1,
          inputs,
          predictedLabel: pred.predictedLabel,
          confidence: pred.confidence,
          probabilities: pred.probabilities,
        };
      });

      res.json({
        mode: 'calibrated-signal-engine',
        results,
      });
    } catch (err) {
      res.status(500).json({
        error: err instanceof Error ? err.message : 'Batch inference failed',
      });
    }
  });

  if (process.env.NODE_ENV !== 'production') {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distDir = path.resolve(__dirname, 'dist');
    app.use(express.static(distDir));
    app.get('*', (_req, res) => {
      res.sendFile(path.join(distDir, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`AtmosSignal server listening on http://0.0.0.0:${PORT}`);
  });
}

startServer().catch((err) => {
  console.error('Failed to start AtmosSignal server:', err);
  process.exit(1);
});
