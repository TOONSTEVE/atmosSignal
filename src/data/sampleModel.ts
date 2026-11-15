import type { FeatureSpec, ModelSchema, WeatherPreset } from '../types/weather';

export const SAMPLE_FEATURES: FeatureSpec[] = [
  {
    key: 'temperature_c',
    label: 'Air Temperature',
    unit: '°C',
    min: -15,
    max: 45,
    step: 0.5,
    defaultValue: 21.5,
    nominalCenter: 20,
    inferredFromFile: false,
  },
  {
    key: 'humidity_pct',
    label: 'Relative Humidity',
    unit: '%',
    min: 10,
    max: 100,
    step: 1,
    defaultValue: 58,
    nominalCenter: 55,
    inferredFromFile: false,
  },
  {
    key: 'pressure_hpa',
    label: 'Barometric Pressure',
    unit: 'hPa',
    min: 970,
    max: 1045,
    step: 0.5,
    defaultValue: 1014.5,
    nominalCenter: 1013.2,
    inferredFromFile: false,
  },
  {
    key: 'wind_speed_kmh',
    label: 'Surface Wind Speed',
    unit: 'km/h',
    min: 0,
    max: 110,
    step: 1,
    defaultValue: 14,
    nominalCenter: 15,
    inferredFromFile: false,
  },
  {
    key: 'cloud_cover_pct',
    label: 'Cloud Cover',
    unit: '%',
    min: 0,
    max: 100,
    step: 1,
    defaultValue: 28,
    nominalCenter: 35,
    inferredFromFile: false,
  },
  {
    key: 'dew_point_c',
    label: 'Dew Point',
    unit: '°C',
    min: -20,
    max: 32,
    step: 0.5,
    defaultValue: 12.5,
    nominalCenter: 11,
    inferredFromFile: false,
  },
];

export const SAMPLE_CLASSES = ['Clear', 'Overcast', 'Rain', 'Storm', 'Fog'];

export const SAMPLE_MODEL_SCHEMA: ModelSchema = {
  id: 'sample-atmospheric-rf',
  source: 'sample',
  modelFileName: 'weather_signal_model.pkl (sample)',
  encoderFileName: 'label_encoder.pkl (sample)',
  estimatorClass: 'RandomForestClassifier',
  estimatorModule: 'sklearn.ensemble._forest',
  serializationFormat: 'joblib (zlib)',
  sklearnVersion: '1.5.1',
  features: SAMPLE_FEATURES,
  classes: SAMPLE_CLASSES,
  hyperparameters: {
    n_estimators: 200,
    max_depth: 12,
    criterion: 'gini',
    random_state: 42,
  },
  pipelineSteps: [
    { name: 'scaler', class: 'StandardScaler' },
    { name: 'classifier', class: 'RandomForestClassifier' },
  ],
  warnings: [],
  modelArtifact: null,
  encoderArtifact: null,
};

export const WEATHER_PRESETS: WeatherPreset[] = [
  {
    id: 'clear-ridge',
    name: 'Clear High-Pressure Ridge',
    summary: 'Dry descending air mass with crisp visibility',
    values: {
      temperature_c: 23.0,
      humidity_pct: 42,
      pressure_hpa: 1024.0,
      wind_speed_kmh: 11,
      cloud_cover_pct: 12,
      dew_point_c: 9.0,
    },
  },
  {
    id: 'frontal-storm',
    name: 'Frontal Thunderstorm Cell',
    summary: 'Deepening surface trough with strong gust front',
    values: {
      temperature_c: 26.5,
      humidity_pct: 86,
      pressure_hpa: 993.5,
      wind_speed_kmh: 58,
      cloud_cover_pct: 94,
      dew_point_c: 23.5,
    },
  },
  {
    id: 'coastal-rain',
    name: 'Maritime Rain Band',
    summary: 'Saturated onshore flow and steady stratiform rain',
    values: {
      temperature_c: 15.5,
      humidity_pct: 91,
      pressure_hpa: 1004.0,
      wind_speed_kmh: 26,
      cloud_cover_pct: 88,
      dew_point_c: 14.0,
    },
  },
  {
    id: 'radiation-fog',
    name: 'Dawn Radiation Fog',
    summary: 'Near-zero dewpoint spread under calm nocturnal boundary',
    values: {
      temperature_c: 9.0,
      humidity_pct: 97,
      pressure_hpa: 1016.0,
      wind_speed_kmh: 3,
      cloud_cover_pct: 45,
      dew_point_c: 8.5,
    },
  },
  {
    id: 'autumn-overcast',
    name: 'Stratocumulus Overcast',
    summary: 'Broad cloud deck with neutral pressure tendency',
    values: {
      temperature_c: 16.0,
      humidity_pct: 72,
      pressure_hpa: 1012.0,
      wind_speed_kmh: 16,
      cloud_cover_pct: 82,
      dew_point_c: 11.0,
    },
  },
];

export function inferFeatureSpec(rawName: string, index: number): FeatureSpec {
  const lower = rawName.toLowerCase();
  const pretty = rawName
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase());

  if (lower.includes('temp') || lower.includes('t_c') || lower.includes('tmax') || lower.includes('tmin')) {
    return {
      key: rawName,
      label: pretty,
      unit: lower.includes('_f') ? '°F' : '°C',
      min: -20,
      max: 48,
      step: 0.5,
      defaultValue: 20,
      nominalCenter: 20,
      inferredFromFile: true,
    };
  }
  if (lower.includes('humid') || lower.includes('rh') || lower.includes('cloud')) {
    return {
      key: rawName,
      label: pretty,
      unit: '%',
      min: 0,
      max: 100,
      step: 1,
      defaultValue: 60,
      nominalCenter: 55,
      inferredFromFile: true,
    };
  }
  if (lower.includes('press') || lower.includes('slp') || lower.includes('baro')) {
    return {
      key: rawName,
      label: pretty,
      unit: 'hPa',
      min: 960,
      max: 1050,
      step: 0.5,
      defaultValue: 1013,
      nominalCenter: 1013,
      inferredFromFile: true,
    };
  }
  if (lower.includes('wind') || lower.includes('gust')) {
    return {
      key: rawName,
      label: pretty,
      unit: 'km/h',
      min: 0,
      max: 120,
      step: 1,
      defaultValue: 15,
      nominalCenter: 15,
      inferredFromFile: true,
    };
  }
  if (lower.includes('dew')) {
    return {
      key: rawName,
      label: pretty,
      unit: '°C',
      min: -25,
      max: 35,
      step: 0.5,
      defaultValue: 11,
      nominalCenter: 11,
      inferredFromFile: true,
    };
  }
  if (lower.includes('rain') || lower.includes('precip')) {
    return {
      key: rawName,
      label: pretty,
      unit: 'mm',
      min: 0,
      max: 150,
      step: 0.5,
      defaultValue: 2,
      nominalCenter: 2,
      inferredFromFile: true,
    };
  }
  if (lower.includes('solar') || lower.includes('rad') || lower.includes('uv')) {
    return {
      key: rawName,
      label: pretty,
      unit: lower.includes('uv') ? 'UV' : 'W/m²',
      min: 0,
      max: lower.includes('uv') ? 14 : 1100,
      step: lower.includes('uv') ? 0.5 : 10,
      defaultValue: lower.includes('uv') ? 4 : 380,
      nominalCenter: lower.includes('uv') ? 4 : 380,
      inferredFromFile: true,
    };
  }

  return {
    key: rawName || `feature_${index}`,
    label: pretty || `Signal ${index + 1}`,
    unit: 'val',
    min: -10,
    max: 100,
    step: 0.5,
    defaultValue: 25,
    nominalCenter: 25,
    inferredFromFile: true,
  };
}

export function generateSampleBatchCsv(features: FeatureSpec[]): string {
  const headers = features.map((f) => f.key).join(',');
  const rows = WEATHER_PRESETS.map((preset, rowIdx) => {
    return features
      .map((f, fIdx) => {
        if (typeof preset.values[f.key] === 'number') {
          return preset.values[f.key].toFixed(1);
        }
        const span = f.max - f.min;
        const val = f.min + span * (0.2 + ((rowIdx * 0.17 + fIdx * 0.13) % 0.65));
        return val.toFixed(1);
      })
      .join(',');
  });
  return [headers, ...rows].join('\n');
}
