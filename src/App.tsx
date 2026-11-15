import { useCallback, useEffect, useState } from 'react';
import { TopBar, type ActiveTab } from './components/TopBar';
import { InspectorView } from './components/InspectorView';
import { ForecastConsole } from './components/ForecastConsole';
import { BatchCsvView } from './components/BatchCsvView';
import { PythonBridgeView } from './components/PythonBridgeView';
import {
  SAMPLE_CLASSES,
  SAMPLE_FEATURES,
  SAMPLE_MODEL_SCHEMA,
  inferFeatureSpec,
} from './data/sampleModel';
import type {
  ForecastResult,
  InspectedArtifact,
  ModelSchema,
  RuntimeCapabilities,
} from './types/weather';

function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = String(reader.result || '');
      const commaIdx = result.indexOf(',');
      resolve(commaIdx >= 0 ? result.slice(commaIdx + 1) : result);
    };
    reader.onerror = () => reject(new Error(`Failed to read ${file.name}`));
    reader.readAsDataURL(file);
  });
}

function buildDefaultsFromSchema(schema: ModelSchema): Record<string, number> {
  const next: Record<string, number> = {};
  for (const f of schema.features) {
    next[f.key] = f.defaultValue;
  }
  return next;
}

export default function App() {
  const [activeTab, setActiveTab] = useState<ActiveTab>('inspector');
  const [capabilities, setCapabilities] = useState<RuntimeCapabilities | null>(null);
  const [schema, setSchema] = useState<ModelSchema | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [inputs, setInputs] = useState<Record<string, number>>({});
  const [sweepFeature, setSweepFeature] = useState<string>('pressure_hpa');
  const [forecast, setForecast] = useState<ForecastResult | null>(null);
  const [isInspecting, setIsInspecting] = useState(false);
  const [isPredicting, setIsPredicting] = useState(false);
  const [inspectError, setInspectError] = useState<string | null>(null);

  useEffect(() => {
    fetch('/api/capabilities')
      .then((res) => res.json())
      .then((data: RuntimeCapabilities) => setCapabilities(data))
      .catch(() => {
        setCapabilities({
          pythonVersion: '3.10',
          hasSklearn: false,
          hasJoblib: false,
          hasNumpy: false,
          hasPandas: false,
          sklearnVersion: null,
          joblibVersion: null,
          canRunNativePklInference: false,
        });
      });
  }, []);

  const applySchema = useCallback((nextSchema: ModelSchema, nextSessionId: string | null) => {
    setSchema(nextSchema);
    setSessionId(nextSessionId);
    const defaults = buildDefaultsFromSchema(nextSchema);
    setInputs(defaults);
    const preferredSweep =
      nextSchema.features.find((f) => f.key.toLowerCase().includes('press'))?.key ||
      nextSchema.features[0]?.key ||
      'pressure_hpa';
    setSweepFeature(preferredSweep);
  }, []);

  const handleLoadSample = useCallback(() => {
    setInspectError(null);
    applySchema(SAMPLE_MODEL_SCHEMA, null);
    setActiveTab('forecast');
  }, [applySchema]);

  const handleInspectFiles = useCallback(
    async (modelFile: File | null, encoderFile: File | null) => {
      setIsInspecting(true);
      setInspectError(null);
      try {
        const [modelBase64, encoderBase64] = await Promise.all([
          modelFile ? fileToBase64(modelFile) : Promise.resolve(undefined),
          encoderFile ? fileToBase64(encoderFile) : Promise.resolve(undefined),
        ]);

        const response = await fetch('/api/inspect-pkl', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            modelFileName: modelFile?.name,
            modelBase64,
            encoderFileName: encoderFile?.name,
            encoderBase64,
          }),
        });

        const data = (await response.json()) as {
          sessionId?: string;
          runtimeCapabilities?: RuntimeCapabilities;
          modelInfo?: InspectedArtifact | null;
          encoderInfo?: InspectedArtifact | null;
          error?: string;
        };

        if (!response.ok || data.error) {
          throw new Error(data.error || 'Failed to inspect .pkl files.');
        }

        if (data.runtimeCapabilities) {
          setCapabilities(data.runtimeCapabilities);
        }

        const mInfo = data.modelInfo ?? null;
        const eInfo = data.encoderInfo ?? null;
        const warnings: string[] = [
          ...(mInfo?.warnings || []),
          ...(eInfo?.warnings || []),
        ];

        const rawFeatureNames = mInfo?.featureNamesIn || [];
        let features = rawFeatureNames.map((name, idx) => inferFeatureSpec(name, idx));

        if (features.length === 0) {
          const count = mInfo?.nFeaturesIn || 0;
          if (count > 0) {
            features = Array.from({ length: count }, (_, idx) =>
              inferFeatureSpec(`signal_feature_${idx}`, idx)
            );
            warnings.push(
              `Model stored n_features_in_=${count} without named columns (likely trained on a NumPy array). You can rename or customize these columns below.`
            );
          } else {
            features = SAMPLE_FEATURES;
            warnings.push(
              'Input column names were not embedded as plain pickle strings (common with compressed joblib numpy buffers). Pre-filled default meteorological columns—customize them below to match your training DataFrame.'
            );
          }
        }

        let classes = eInfo?.classes?.length
          ? eInfo.classes
          : mInfo?.classes?.length
            ? mInfo.classes
            : [];

        if (classes.length === 0) {
          classes = SAMPLE_CLASSES;
          warnings.push(
            'Target label strings were not readable without numpy/scikit-learn unpickling. Using default weather classes—edit or add your exact LabelEncoder labels below.'
          );
        }

        const nextSchema: ModelSchema = {
          id: data.sessionId || `pkl-${Date.now()}`,
          source: data.runtimeCapabilities?.canRunNativePklInference
            ? 'live-sklearn'
            : 'inspected-pkl',
          modelFileName: mInfo?.fileName || modelFile?.name || 'weather_signal_model.pkl',
          encoderFileName: eInfo?.fileName || encoderFile?.name || null,
          estimatorClass: mInfo?.estimatorClass || 'ScikitLearnEstimator',
          estimatorModule: mInfo?.estimatorModule || 'sklearn',
          serializationFormat: mInfo?.serializationFormat || eInfo?.serializationFormat || 'pickle',
          sklearnVersion: mInfo?.sklearnVersion || eInfo?.sklearnVersion || 'Unknown',
          features,
          classes,
          hyperparameters: mInfo?.hyperparameters || {},
          pipelineSteps: mInfo?.pipelineSteps || [],
          warnings,
          modelArtifact: mInfo,
          encoderArtifact: eInfo,
        };

        applySchema(nextSchema, data.sessionId ?? null);
      } catch (err) {
        setInspectError(
          err instanceof Error ? err.message : 'Unable to inspect uploaded .pkl files.'
        );
      } finally {
        setIsInspecting(false);
      }
    },
    [applySchema]
  );

  const handleLoadJsonFile = useCallback(
    async (jsonFile: File) => {
      setIsInspecting(true);
      setInspectError(null);
      try {
        const text = await jsonFile.text();
        const parsed = JSON.parse(text) as Record<string, unknown>;

        const response = await fetch('/api/load-json-model', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(parsed),
        });

        const data = (await response.json()) as {
          sessionId?: string;
          estimatorClass?: string;
          estimatorModule?: string;
          sklearnVersion?: string;
          modelFileName?: string;
          encoderFileName?: string;
          features?: string[];
          classes?: string[];
          hyperparameters?: Record<string, string | number | boolean>;
          pipelineSteps?: Array<{ name: string; class: string }>;
          error?: string;
        };

        if (!response.ok || data.error) {
          throw new Error(data.error || 'Failed to load exported JSON bundle.');
        }

        const features = (data.features || []).map((name, idx) =>
          inferFeatureSpec(name, idx)
        );

        const nextSchema: ModelSchema = {
          id: data.sessionId || `json-${Date.now()}`,
          source: 'exported-json',
          modelFileName: data.modelFileName || jsonFile.name,
          encoderFileName: data.encoderFileName || null,
          estimatorClass: data.estimatorClass || 'SklearnModel',
          estimatorModule: data.estimatorModule || 'sklearn',
          serializationFormat: 'atmossignal-sklearn-v1 (JSON)',
          sklearnVersion: data.sklearnVersion || 'exported',
          features,
          classes: data.classes || SAMPLE_CLASSES,
          hyperparameters: data.hyperparameters || {},
          pipelineSteps: data.pipelineSteps || [],
          warnings: [],
          modelArtifact: null,
          encoderArtifact: null,
        };

        applySchema(nextSchema, data.sessionId ?? null);
        setActiveTab('forecast');
      } catch (err) {
        setInspectError(
          err instanceof Error ? err.message : 'Invalid exported model JSON file.'
        );
      } finally {
        setIsInspecting(false);
      }
    },
    [applySchema]
  );

  // Trigger live prediction whenever schema, inputs, or sweepFeature changes
  useEffect(() => {
    if (!schema) {
      setForecast(null);
      return;
    }

    const activeSweepSpec =
      schema.features.find((f) => f.key === sweepFeature) || schema.features[0];

    const timer = setTimeout(async () => {
      setIsPredicting(true);
      try {
        const response = await fetch('/api/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            sessionId,
            featureOrder: schema.features.map((f) => f.key),
            classes: schema.classes,
            inputs,
            sweepFeature: activeSweepSpec?.key,
            sweepMin: activeSweepSpec?.min ?? 0,
            sweepMax: activeSweepSpec?.max ?? 100,
          }),
        });
        if (response.ok) {
          const data = (await response.json()) as ForecastResult;
          setForecast(data);
        }
      } catch {
        // Keep previous forecast on transient network cancel
      } finally {
        setIsPredicting(false);
      }
    }, 60);

    return () => clearTimeout(timer);
  }, [schema, sessionId, inputs, sweepFeature]);

  const handleChangeInput = (key: string, value: number) => {
    setInputs((prev) => ({ ...prev, [key]: value }));
  };

  const handleApplyPreset = (presetValues: Record<string, number>) => {
    if (!schema) return;
    setInputs((prev) => {
      const next = { ...prev };
      for (const f of schema.features) {
        if (typeof presetValues[f.key] === 'number') {
          next[f.key] = presetValues[f.key];
        }
      }
      return next;
    });
  };

  const handleResetNominal = () => {
    if (!schema) return;
    setInputs(buildDefaultsFromSchema(schema));
  };

  const handleUpdateSchema = (updated: ModelSchema) => {
    setSchema(updated);
    setInputs((prev) => {
      const next: Record<string, number> = {};
      for (const f of updated.features) {
        next[f.key] = typeof prev[f.key] === 'number' ? prev[f.key] : f.defaultValue;
      }
      return next;
    });
    if (!updated.features.some((f) => f.key === sweepFeature) && updated.features[0]) {
      setSweepFeature(updated.features[0].key);
    }
  };

  return (
    <div className="min-h-screen bg-[#07090E] text-slate-100 flex flex-col">
      <TopBar
        activeTab={activeTab}
        onSelectTab={setActiveTab}
        onQuickUploadClick={() => setActiveTab('inspector')}
      />

      <div className="flex-1">
        {activeTab === 'inspector' && (
          <InspectorView
            schema={schema}
            capabilities={capabilities}
            isInspecting={isInspecting}
            inspectError={inspectError}
            onInspectFiles={handleInspectFiles}
            onLoadJsonFile={handleLoadJsonFile}
            onLoadSample={handleLoadSample}
            onUpdateSchema={handleUpdateSchema}
            onNavigate={(tab) => setActiveTab(tab)}
          />
        )}

        {activeTab === 'forecast' && (
          <ForecastConsole
            schema={schema}
            inputs={inputs}
            forecast={forecast}
            isPredicting={isPredicting}
            sweepFeature={sweepFeature}
            onChangeInput={handleChangeInput}
            onApplyPreset={handleApplyPreset}
            onResetNominal={handleResetNominal}
            onChangeSweepFeature={setSweepFeature}
            onLoadSample={handleLoadSample}
          />
        )}

        {activeTab === 'batch' && (
          <BatchCsvView
            schema={schema}
            sessionId={sessionId}
            onLoadSample={handleLoadSample}
          />
        )}

        {activeTab === 'bridge' && <PythonBridgeView schema={schema} />}
      </div>
    </div>
  );
}
