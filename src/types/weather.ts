export interface RuntimeCapabilities {
  pythonVersion: string;
  hasSklearn: boolean;
  hasJoblib: boolean;
  hasNumpy: boolean;
  hasPandas: boolean;
  sklearnVersion: string | null;
  joblibVersion: string | null;
  canRunNativePklInference: boolean;
}

export interface PipelineStep {
  name: string;
  class: string;
}

export interface InspectedArtifact {
  fileName: string;
  sizeBytes: number;
  serializationFormat: string;
  isJoblib: boolean;
  inspectionMethod: 'native-sklearn' | 'pickletools-stub' | 'opcode-scan';
  estimatorClass: string | null;
  estimatorModule: string | null;
  sklearnVersion: string | null;
  nFeaturesIn: number | null;
  featureNamesIn: string[];
  classes: string[];
  hyperparameters: Record<string, string | number | boolean>;
  pipelineSteps: PipelineStep[];
  discoveredModules: string[];
  warnings: string[];
}

export interface FeatureSpec {
  key: string;
  label: string;
  unit: string;
  min: number;
  max: number;
  step: number;
  defaultValue: number;
  nominalCenter: number;
  inferredFromFile: boolean;
}

export interface ModelSchema {
  id: string;
  source: 'sample' | 'inspected-pkl' | 'live-sklearn' | 'exported-json';
  modelFileName: string;
  encoderFileName: string | null;
  estimatorClass: string;
  estimatorModule: string;
  serializationFormat: string;
  sklearnVersion: string;
  features: FeatureSpec[];
  classes: string[];
  hyperparameters: Record<string, string | number | boolean>;
  pipelineSteps: PipelineStep[];
  warnings: string[];
  modelArtifact: InspectedArtifact | null;
  encoderArtifact: InspectedArtifact | null;
}

export interface ClassProbability {
  label: string;
  probability: number;
}

export interface SensitivityPoint {
  xValue: number;
  probabilities: Record<string, number>;
}

export interface ForecastResult {
  mode: 'live-sklearn' | 'native-json-model' | 'calibrated-signal-engine';
  predictedLabel: string;
  confidence: number;
  probabilities: ClassProbability[];
  primarySweepFeature: string;
  sensitivitySeries: SensitivityPoint[];
  advisoryNote: string;
}

export interface BatchRowResult {
  rowIndex: number;
  inputs: Record<string, number>;
  predictedLabel: string;
  confidence: number;
  probabilities: ClassProbability[];
}

export interface WeatherPreset {
  id: string;
  name: string;
  summary: string;
  values: Record<string, number>;
}
