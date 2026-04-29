export type NumberFormatMode = 'compact' | 'full';

interface FormatNumberOptions {
  mode?: NumberFormatMode;
  locale?: string;
  maximumFractionDigits?: number;
  minimumFractionDigits?: number;
  fallback?: string;
}

interface FormatPercentOptions {
  locale?: string;
  input?: 'ratio' | 'percent';
  maximumFractionDigits?: number;
  fallback?: string;
}

const DEFAULT_FALLBACK = '-';

function toFiniteNumber(value: unknown): number | null {
  if (typeof value === 'number') {
    return Number.isFinite(value) ? value : null;
  }
  if (typeof value === 'string') {
    const cleaned = value.trim().replace(/,/g, '');
    if (cleaned === '') return null;
    const parsed = Number(cleaned);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

function trimZeros(value: string) {
  return value.replace(/(\.\d*?[1-9])0+$/u, '$1').replace(/\.0+$/u, '');
}

function decimalString(value: number, fractionDigits: number) {
  return trimZeros(value.toFixed(fractionDigits));
}

export function formatNumber(value: unknown, options: FormatNumberOptions = {}) {
  const {
    mode = 'compact',
    locale = 'en-US',
    maximumFractionDigits,
    minimumFractionDigits,
    fallback = DEFAULT_FALLBACK,
  } = options;

  const numeric = toFiniteNumber(value);
  if (numeric === null) return fallback;

  if (mode === 'compact') {
    const absValue = Math.abs(numeric);
    const sign = numeric < 0 ? '-' : '';

    if (absValue >= 1_000_000_000) {
      const scaled = absValue / 1_000_000_000;
      const suffix = locale.toLowerCase().startsWith('tr') ? ' milyar' : 'B';
      return `${sign}${decimalString(scaled, maximumFractionDigits ?? 1)}${suffix}`;
    }
    if (absValue >= 1_000_000) {
      return `${sign}${decimalString(absValue / 1_000_000, maximumFractionDigits ?? 1)}M`;
    }
    if (absValue >= 1_000) {
      return `${sign}${decimalString(absValue / 1_000, maximumFractionDigits ?? 1)}K`;
    }
  }

  return new Intl.NumberFormat(locale, {
    maximumFractionDigits: maximumFractionDigits ?? 2,
    minimumFractionDigits,
  }).format(numeric);
}

export function formatPercent(value: unknown, options: FormatPercentOptions = {}) {
  const {
    locale = 'en-US',
    input = 'ratio',
    maximumFractionDigits = 1,
    fallback = DEFAULT_FALLBACK,
  } = options;
  const numeric = toFiniteNumber(value);
  if (numeric === null) return fallback;
  const percent = input === 'ratio' ? numeric * 100 : numeric;
  return `${new Intl.NumberFormat(locale, { maximumFractionDigits }).format(percent)}%`;
}

export function compactLabel(value: unknown, maxLength = 18) {
  const text = String(value ?? '').trim();
  if (text.length <= maxLength) return text || DEFAULT_FALLBACK;
  return `${text.slice(0, Math.max(0, maxLength - 3)).trimEnd()}...`;
}
