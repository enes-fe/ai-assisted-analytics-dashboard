import { useState, useEffect, useRef } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, Legend, ResponsiveContainer,
  LineChart, Line, PieChart, Pie, Cell, AreaChart, Area, ScatterChart, Scatter, ComposedChart, ReferenceLine
} from 'recharts';
import { MoreHorizontal, Info, AlertTriangle, ShieldAlert, X } from 'lucide-react';
import { useLang } from '../contexts/useLang';
import './ChartWidget.css';

const Hexagon = (props: any) => {
  const { cx, cy, fill } = props;
  return (
    <path
      d={`M${cx},${cy - 6} L${cx + 5},${cy - 3} L${cx + 5},${cy + 3} L${cx},${cy + 6} L${cx - 5},${cy + 3} L${cx - 5},${cy - 3} Z`}
      fill={fill} fillOpacity={0.6}
    />
  );
};

export interface ChartConfig {
  id: string;
  type: 'bar' | 'line' | 'pie' | 'donut' | 'area' | 'scatter' | 'table' | 'forecast' | 'clustering' | 'boxplot';
  title: string;
  xAxisKey: string;
  series: { key: string; color?: string; name?: string }[];
  insight?: string;
  isPredictive?: boolean;
  isHexbin?: boolean;
  layout?: 'horizontal' | 'vertical';
  chartData?: any[];
  historical?: any[];
  forecast?: any[];
  warnings?: string[];
  cluster_profiles?: any[];
  scatter_col_x?: string;
  scatter_col_y?: string;
  xAxisLabel?: string;
  yAxisLabel?: string;
  projection_note?: string;
  projection_method?: string;
  aggregation_method?: string;
  aggregation_label?: string;
  silhouette_score?: number | null;
  silhouette_quality?: string | null;
  silhouette_text?: string;
  labelKey?: string;
  labelName?: string;
  metricKeys?: string[];
  showRegressionLine?: boolean;
  rSquared?: number;
  regressionCoeffs?: { slope: number; intercept: number };
  isNormal?: boolean;
  normalityNote?: string;
  mean?: number;
  median?: number;
  isStacked?: boolean;
  cramersV?: number;
  quality?: string;
  effect_size?: number;
  effect_label?: string;
}

interface ChartWidgetProps {
  config: ChartConfig;
  data: any[];
  numberFormat?: 'compact' | 'full';
  onRemove?: () => void;
  exportMode?: boolean;
}

const COLORS = [
  '#6366f1', '#10b981', '#f59e0b', '#ef4444', '#06b6d4',
  '#8b5cf6', '#ec4899', '#64748b', '#0ea5e9', '#f97316'
];

const VIBRANT_GRADIENT = [
  '#6366f1', '#818cf8', '#10b981', '#34d399', '#f59e0b', '#fbbf24',
  '#06b6d4', '#22d3ee', '#8b5cf6', '#a78bfa', '#ec4899', '#f472b6',
  '#ef4444', '#f87171', '#64748b', '#94a3b8'
];

export default function ChartWidget({ config, data, numberFormat = 'compact', onRemove, exportMode = false }: ChartWidgetProps) {
  const chartDataToUse = config.chartData || data;
  const storageKey = `chart_title_${config.id}`;
  const typeKey = `chart_type_${config.id}`;
  const insightStorageKey = `chart_insight_${config.id}`;

  // Chart type that can be overridden by the user (persisted)
  const SWITCHABLE: Array<ChartConfig['type']> = ['bar', 'line', 'area'];
  const canSwitch = SWITCHABLE.includes(config.type as any);

  const [editableTitle, setEditableTitle] = useState(
    () => localStorage.getItem(storageKey) || config.title
  );
  const [activeType, setActiveType] = useState<ChartConfig['type']>(
    () => (localStorage.getItem(typeKey) as ChartConfig['type']) || config.type
  );
  const [editableInsight, setEditableInsight] = useState(() => {
    const savedInsight = localStorage.getItem(insightStorageKey);
    return savedInsight !== null ? savedInsight : config.insight || '';
  });
  const [lockedPoint, setLockedPoint] = useState<any | null>(null);
  const [localFormat, setLocalFormat] = useState<'compact' | 'full' | null>(null);
  const [showMenu, setShowMenu] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);
  const { t } = useLang();
  const c = t.chart;

  // Persist title to localStorage on change
  const handleTitleChange = (val: string) => {
    setEditableTitle(val);
    localStorage.setItem(storageKey, val);
  };

  const handleInsightChange = (val: string) => {
    setEditableInsight(val);
    localStorage.setItem(insightStorageKey, val);
  };

  const handleTypeSwitch = (type: ChartConfig['type']) => {
    setActiveType(type);
    localStorage.setItem(typeKey, type);
    setShowMenu(false);
  };

  useEffect(() => { setLocalFormat(null); }, [numberFormat]);
  useEffect(() => { setLockedPoint(null); }, [config.id, activeType]);



  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) setShowMenu(false);
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const activeFormat = localFormat || numberFormat;
  const hasVisibleInsight = editableInsight.trim().length > 0;

  const yAxisFormatter = (value: number) => {
    if (activeFormat === 'compact') {
      return new Intl.NumberFormat('en-US', { notation: 'compact', maximumFractionDigits: 1 }).format(value);
    }
    return new Intl.NumberFormat('en-US', { maximumFractionDigits: 1 }).format(value);
  };

  const tooltipFormatter = (value: any) => {
    if (typeof value === 'number') {
      if (activeFormat === 'compact') {
        return new Intl.NumberFormat('en-US', { notation: 'compact', maximumFractionDigits: 2 }).format(value);
      }
      return new Intl.NumberFormat('en-US', { maximumFractionDigits: 2 }).format(value);
    }
    return value;
  };

  const rowLabel = (row: any) => {
    const key = config.labelKey || '__label';
    const explicit = row?.[key] || row?.cluster_name || row?.name || row?.Name || row?.label;
    if (explicit) return explicit;
    const entityKey = Object.keys(row || {}).find(k => {
      const lower = k.toLowerCase();
      if (['cluster', 'cluster_name', '__row'].includes(lower)) return false;
      if (typeof row[k] !== 'string') return false;
      return ['player', 'oyuncu', 'name', 'isim', 'product', 'urun', 'team', 'club', 'customer'].some(token => lower.includes(token));
    });
    if (entityKey) return row[entityKey];
    return row?.__row ? `${c.row || 'Satır'} ${row.__row}` : null;
  };

  const renderPointTooltip = ({ active, payload }: any) => {
    if (!active || !payload?.length) return null;
    const row = payload[0]?.payload;
    if (!row) return null;
    const label = rowLabel(row);
    const xKey = config.type === 'clustering' ? config.scatter_col_x : config.xAxisKey;
    const yKey = config.type === 'clustering' ? config.scatter_col_y : config.series[0]?.key;
    const extraKeys = (config.metricKeys || []).filter(key => key !== xKey && key !== yKey && row[key] !== undefined).slice(0, 6);
    return (
      <div className="chart-tooltip">
        {label && <div className="chart-tooltip-title">{label}</div>}
        {row.cluster_name && <div>{c.cluster}: <b>{row.cluster_name}</b></div>}
        {xKey && <div>{xKey}: <b>{tooltipFormatter(row[xKey])}</b></div>}
        {yKey && <div>{yKey}: <b>{tooltipFormatter(row[yKey])}</b></div>}
        {extraKeys.map(key => (
          <div key={key}>{key}: <b>{tooltipFormatter(row[key])}</b></div>
        ))}
      </div>
    );
  };

  const getNearbyPoints = (row: any, xKey?: string, yKey?: string) => {
    if (!row || !xKey || !yKey) return row ? [row] : [];
    const rows = Array.isArray(chartDataToUse) ? chartDataToUse : [];
    const x = Number(row[xKey]);
    const y = Number(row[yKey]);
    if (!Number.isFinite(x) || !Number.isFinite(y)) return [row];

    const numericRows = rows.filter((item: any) => Number.isFinite(Number(item?.[xKey])) && Number.isFinite(Number(item?.[yKey])));
    const xVals = numericRows.map((item: any) => Number(item[xKey]));
    const yVals = numericRows.map((item: any) => Number(item[yKey]));
    const xRange = Math.max(...xVals) - Math.min(...xVals) || 1;
    const yRange = Math.max(...yVals) - Math.min(...yVals) || 1;

    return numericRows
      .map((item: any) => {
        const dx = Math.abs(Number(item[xKey]) - x) / xRange;
        const dy = Math.abs(Number(item[yKey]) - y) / yRange;
        return { item, distance: Math.sqrt(dx * dx + dy * dy), sameX: dx < 0.002 };
      })
      .filter(({ distance, sameX }) => distance <= 0.045 || (sameX && distance <= 0.12))
      .sort((a, b) => a.distance - b.distance)
      .slice(0, 10)
      .map(({ item }) => item);
  };

  const renderScatterShape = (props: any) => {
    const { cx, cy, fill } = props;
    if (!Number.isFinite(cx) || !Number.isFinite(cy)) return null;
    return (
      <g className="scatter-point-hit">
        <circle className="scatter-hit-area" cx={cx} cy={cy} r={14} />
        {config.isHexbin ? <Hexagon cx={cx} cy={cy} fill={fill} /> : <circle cx={cx} cy={cy} r={5} fill={fill} fillOpacity={0.72} />}
      </g>
    );
  };

  const renderLockedPointPanel = (xKey?: string, yKey?: string) => {
    if (!lockedPoint || !xKey || !yKey) return null;
    const nearby = getNearbyPoints(lockedPoint, xKey, yKey);
    const keys = Array.from(new Set([xKey, yKey, ...(config.metricKeys || [])])).filter(Boolean) as string[];
    return (
      <div className="scatter-detail-panel">
        <div className="scatter-detail-header">
          <div>
            <div className="scatter-detail-title">{rowLabel(lockedPoint) || 'Selected point'}</div>
            <div className="scatter-detail-subtitle">{nearby.length > 1 ? `${nearby.length} nearby points` : '1 point'}</div>
          </div>
          <button className="icon-btn-ghost" onClick={() => setLockedPoint(null)} title="Close">
            <X size={14} />
          </button>
        </div>
        <div className="scatter-nearby-list">
          {nearby.map((item: any, index: number) => (
            <div key={`${rowLabel(item) || item.__row || index}-${index}`} className="scatter-nearby-row">
              <div className="scatter-nearby-name">{rowLabel(item) || `${c.row || 'Row'} ${item.__row || index + 1}`}</div>
              <div className="scatter-nearby-values">
                {keys.slice(0, 5).map(key => (
                  item[key] !== undefined ? <span key={key}>{key}: <b>{tooltipFormatter(item[key])}</b></span> : null
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  };

  const renderChart = () => {
    switch (activeType) {

      case 'bar': {
        const barData = (chartDataToUse as any[]).slice(0, 5);
        const categoryLabels = barData.map(row => String(row?.[config.xAxisKey] ?? ''));
        const maxLabelLength = Math.max(0, ...categoryLabels.map(label => label.length));
        const isVertical = config.layout === 'vertical' || categoryLabels.length > 8 || maxLabelLength > 14;
        const yAxisWidth = Math.min(190, Math.max(90, maxLabelLength * 7));
        const chartHeight = Math.min(560, Math.max(320, categoryLabels.length * 30 + 90));
        return (
          <ResponsiveContainer width="100%" height={isVertical ? chartHeight : 300} debounce={50}>
            <BarChart data={barData} layout={isVertical ? 'vertical' : 'horizontal'} margin={{ top: 10, right: 30, left: isVertical ? 4 : -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--border-color)" />
              {isVertical ? (
                <>
                  <XAxis type="number" stroke="var(--text-muted)" fontSize={12} tickLine={false} axisLine={false} tickFormatter={yAxisFormatter} domain={config.isStacked ? [0, 100] : [0, 'auto']} />
                  <YAxis type="category" dataKey={config.xAxisKey} stroke="var(--text-muted)" fontSize={11} tickLine={false} axisLine={false} width={yAxisWidth} tickFormatter={(val) => String(val).length > 24 ? String(val).substring(0, 22) + '...' : val} />
                </>
              ) : (
                <>
                  <XAxis dataKey={config.xAxisKey} stroke="var(--text-muted)" fontSize={11} tickLine={false} axisLine={false} interval={0} tickFormatter={(val) => String(val).length > 12 ? String(val).substring(0, 10) + '...' : val} />
                  <YAxis stroke="var(--text-muted)" fontSize={12} tickLine={false} axisLine={false} tickFormatter={yAxisFormatter} domain={config.isStacked ? [0, 100] : [0, 'auto']} />
                </>
              )}
              <RechartsTooltip formatter={tooltipFormatter} contentStyle={{ backgroundColor: 'var(--bg-surface)', borderColor: 'var(--border-color)', color: 'var(--text-primary)', borderRadius: '8px', boxShadow: 'var(--shadow-md)' }} />
              <Legend wrapperStyle={{ fontSize: '12px' }} />
              {config.series.map((s, idx) => (
                <Bar key={s.key} dataKey={s.key} name={s.name || s.key} fill={VIBRANT_GRADIENT[idx % VIBRANT_GRADIENT.length]} radius={isVertical ? [0, 4, 4, 0] : [4, 4, 0, 0]} barSize={config.isStacked ? undefined : 20} stackId={config.isStacked ? 'a' : undefined} isAnimationActive={false}>
                  {!config.isStacked && config.series.length === 1 && barData.map((_: any, index: number) => (
                    <Cell key={`cell-${index}`} fill={VIBRANT_GRADIENT[index % VIBRANT_GRADIENT.length]} />
                  ))}
                </Bar>
              ))}
              {config.id.startsWith('hist-') && (
                <>
                  {config.isNormal
                    ? <ReferenceLine x={config.mean} stroke="#10b981" strokeDasharray="3 3" label={{ value: c.mean, position: 'top', fill: '#10b981', fontSize: 10 }} />
                    : <ReferenceLine x={config.median} stroke="#f59e0b" strokeDasharray="3 3" label={{ value: c.median, position: 'top', fill: '#f59e0b', fontSize: 10 }} />
                  }
                </>
              )}
            </BarChart>
          </ResponsiveContainer>
        );
      }
      case 'line':
        return (
          <ResponsiveContainer width="100%" height={300} debounce={50}>
            <LineChart data={chartDataToUse} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--border-color)" />
              <XAxis dataKey={config.xAxisKey} stroke="var(--text-muted)" fontSize={12} tickLine={false} axisLine={false} />
              <YAxis stroke="var(--text-muted)" fontSize={12} tickLine={false} axisLine={false} tickFormatter={yAxisFormatter} />
              <RechartsTooltip formatter={tooltipFormatter} contentStyle={{ backgroundColor: 'var(--bg-surface)', borderColor: 'var(--border-color)', color: 'var(--text-primary)', borderRadius: '8px', boxShadow: 'var(--shadow-md)' }} />
              <Legend wrapperStyle={{ fontSize: '12px' }} />
              {config.series.map((s, idx) => (
                <Line type="monotone" key={s.key} dataKey={s.key} name={s.name || s.key} stroke={s.color || COLORS[idx % COLORS.length]} strokeWidth={3} dot={{ r: 4, strokeWidth: 2 }} activeDot={{ r: 6 }} strokeDasharray={config.isPredictive ? '5 5' : undefined} isAnimationActive={false} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        );
      case 'area':
        return (
          <ResponsiveContainer width="100%" height={300} debounce={50}>
            <AreaChart data={chartDataToUse} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--border-color)" />
              <XAxis dataKey={config.xAxisKey} stroke="var(--text-muted)" fontSize={12} tickLine={false} axisLine={false} />
              <YAxis stroke="var(--text-muted)" fontSize={12} tickLine={false} axisLine={false} tickFormatter={yAxisFormatter} />
              <RechartsTooltip formatter={tooltipFormatter} contentStyle={{ backgroundColor: 'var(--bg-surface)', borderColor: 'var(--border-color)', color: 'var(--text-primary)', borderRadius: '8px', boxShadow: 'var(--shadow-md)' }} />
              <Legend wrapperStyle={{ fontSize: '12px' }} />
              {config.series.map((s, idx) => (
                <Area type="monotone" key={s.key} dataKey={s.key} name={s.name || s.key} fill={s.color || COLORS[idx % COLORS.length]} stroke={s.color || COLORS[idx % COLORS.length]} fillOpacity={0.3} isAnimationActive={false} />
              ))}
            </AreaChart>
          </ResponsiveContainer>
        );
      case 'pie':
      case 'donut': {
        const pieDataKey = config.series[0]?.key;
        return (
          <ResponsiveContainer width="100%" height={300} debounce={50}>
            <PieChart>
              <RechartsTooltip formatter={tooltipFormatter} contentStyle={{ backgroundColor: 'var(--bg-surface)', borderColor: 'var(--border-color)', color: 'var(--text-primary)', borderRadius: '8px', boxShadow: 'var(--shadow-md)' }} />
              <Legend wrapperStyle={{ fontSize: '12px' }} />
              <Pie data={chartDataToUse} dataKey={pieDataKey} nameKey={config.xAxisKey} cx="50%" cy="50%" innerRadius={activeType === 'donut' ? 48 : 0} outerRadius={100} fill="#8884d8" label isAnimationActive={false}>
                {chartDataToUse.map((_: any, index: number) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
            </PieChart>
          </ResponsiveContainer>
        );
      }
      case 'scatter': {
        const scatterXKey = config.xAxisKey;
        const scatterYKey = config.series[0].key;
        let regressionData = null;
        if (config.showRegressionLine && config.regressionCoeffs) {
          const xValues = (chartDataToUse as any[]).map(d => d[scatterXKey]).filter(v => typeof v === 'number');
          if (xValues.length > 0) {
            const minX = Math.min(...xValues); const maxX = Math.max(...xValues);
            const { slope, intercept } = config.regressionCoeffs;
            regressionData = [{ [scatterXKey]: minX, regression: slope * minX + intercept }, { [scatterXKey]: maxX, regression: slope * maxX + intercept }];
          }
        }
        return (
          <>
            <ResponsiveContainer width="100%" height={300} debounce={50}>
              <ComposedChart margin={{ top: 20, right: 20, bottom: 20, left: -20 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--border-color)" />
                <XAxis type="number" dataKey={scatterXKey} name={scatterXKey} stroke="var(--text-muted)" fontSize={12} tickFormatter={yAxisFormatter} domain={['auto', 'auto']} />
                <YAxis type="number" dataKey={scatterYKey} name={scatterYKey} stroke="var(--text-muted)" fontSize={12} tickFormatter={yAxisFormatter} domain={['auto', 'auto']} />
                <RechartsTooltip cursor={{ strokeDasharray: '3 3' }} content={renderPointTooltip} />
                <Scatter name={config.title} data={chartDataToUse} fill="var(--accent)" shape={renderScatterShape} onClick={exportMode ? undefined : (point: any) => setLockedPoint(point?.payload || point)} isAnimationActive={false} />
                {regressionData && (
                  <Line data={regressionData} type="monotone" dataKey="regression" stroke="#ef4444" strokeWidth={2} dot={false} activeDot={false} legendType="none" isAnimationActive={false} />
                )}
              </ComposedChart>
            </ResponsiveContainer>
            {renderLockedPointPanel(scatterXKey, scatterYKey)}
          </>
        );
      }
      case 'forecast': {
        const combinedData = [
          ...(config.historical || []).map(d => ({ ...d, type: 'historical' })),
          ...(config.forecast || []).map(d => ({ ...d, type: 'forecast' }))
        ];
        return (
          <div className="forecast-chart-container">
            <ResponsiveContainer width="100%" height={300} debounce={50}>
              <ComposedChart data={combinedData} margin={{ top: 10, right: 30, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--border-color)" />
                <XAxis dataKey="date" stroke="var(--text-muted)" fontSize={11} tickLine={false} axisLine={false} />
                <YAxis stroke="var(--text-muted)" fontSize={12} tickLine={false} axisLine={false} tickFormatter={yAxisFormatter} />
                <RechartsTooltip contentStyle={{ backgroundColor: 'var(--bg-surface)', borderColor: 'var(--border-color)', borderRadius: '8px' }} />
                <Legend wrapperStyle={{ fontSize: '12px' }} />
                <Line type="monotone" dataKey="actual" name={c.legendActual || c.actual || 'Actual'} stroke="#2563eb" strokeWidth={3} dot={{ r: 4 }} isAnimationActive={false} />
                <Line type="monotone" dataKey="fitted" name={c.legendFit || c.modelFit || 'Model Fit'} stroke="#93c5fd" strokeWidth={2} strokeDasharray="5 5" dot={false} isAnimationActive={false} />
                <Line type="monotone" dataKey="forecast" name={c.legendForecast || c.forecast || 'Forecast'} stroke="#10b981" strokeWidth={3} strokeDasharray="5 5" dot={{ r: 5 }} isAnimationActive={false} />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        );
      }
      case 'boxplot': {
        const boxMin = Math.min(...chartDataToUse.map((d: any) => d.min ?? Infinity));
        const boxMax = Math.max(...chartDataToUse.map((d: any) => d.max ?? -Infinity));
        const rangePad = (boxMax - boxMin) * 0.15 || 1;
        const BoxplotLayer = (props: any) => {
          const { xAxisMap, yAxisMap, data: bpData } = props;
          if (!xAxisMap || !yAxisMap || !bpData) return null;
          const xAxis = Object.values(xAxisMap)[0] as any;
          const yAxis = Object.values(yAxisMap)[0] as any;
          if (!xAxis || !yAxis) return null;
          const { bandSize, scale: xScale } = xAxis;
          const { scale: yScale } = yAxis;
          const bw = Math.max(20, (bandSize || 40) * 0.5);
          return (
            <g>
              {bpData.map((entry: any, i: number) => {
                const xCenter = xScale(entry[config.xAxisKey] ?? entry.group) + (bandSize || 40) / 2;
                const yMin = yScale(entry.min); const yMax = yScale(entry.max);
                const yQ1 = yScale(entry.q1); const yQ3 = yScale(entry.q3);
                const yMedian = yScale(entry.median);
                if ([yMin, yMax, yQ1, yQ3, yMedian].some(isNaN)) return null;
                return (
                  <g key={i}>
                    <line x1={xCenter} y1={yMax} x2={xCenter} y2={yMin} stroke="var(--text-muted)" strokeWidth={1.5} strokeDasharray="4 3" />
                    <line x1={xCenter - bw / 3} y1={yMax} x2={xCenter + bw / 3} y2={yMax} stroke="var(--text-muted)" strokeWidth={1.5} />
                    <line x1={xCenter - bw / 3} y1={yMin} x2={xCenter + bw / 3} y2={yMin} stroke="var(--text-muted)" strokeWidth={1.5} />
                    <rect x={xCenter - bw / 2} y={Math.min(yQ1, yQ3)} width={bw} height={Math.abs(yQ3 - yQ1)} fill="var(--accent)" fillOpacity={0.5} stroke="var(--accent)" strokeWidth={1.5} rx={3} />
                    <line x1={xCenter - bw / 2} y1={yMedian} x2={xCenter + bw / 2} y2={yMedian} stroke="white" strokeWidth={2.5} />
                  </g>
                );
              })}
            </g>
          );
        };
        return (
          <ResponsiveContainer width="100%" height={320} debounce={50}>
            <BarChart data={chartDataToUse} margin={{ top: 20, right: 30, bottom: 20, left: -20 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--border-color)" />
              <XAxis dataKey={config.xAxisKey} stroke="var(--text-muted)" fontSize={11} tickLine={false} axisLine={false} />
              <YAxis stroke="var(--text-muted)" fontSize={12} tickLine={false} axisLine={false} tickFormatter={yAxisFormatter} domain={[boxMin - rangePad, boxMax + rangePad]} />
              <RechartsTooltip contentStyle={{ backgroundColor: 'var(--bg-surface)', borderColor: 'var(--border-color)', borderRadius: '8px' }} cursor={{ fill: 'rgba(255,255,255,0.03)' }}
                content={({ active, payload }) => {
                  if (!active || !payload?.length) return null;
                  const d = payload[0]?.payload;
                  if (!d) return null;
                  return (
                    <div style={{ background: 'var(--bg-surface)', border: '1px solid var(--border-color)', borderRadius: '8px', padding: '10px 14px', fontSize: '12px', lineHeight: 1.8 }}>
                      <div style={{ fontWeight: 600, marginBottom: 4 }}>{d[config.xAxisKey] ?? d.group}</div>
                      <div>{c.max}: <b>{yAxisFormatter(d.max)}</b></div>
                      <div>Q3: <b>{yAxisFormatter(d.q3)}</b></div>
                      <div>{c.median}: <b>{yAxisFormatter(d.median)}</b></div>
                      <div>Q1: <b>{yAxisFormatter(d.q1)}</b></div>
                      <div>{c.min}: <b>{yAxisFormatter(d.min)}</b></div>
                    </div>
                  );
                }}
              />
              <Bar dataKey="median" fill="transparent" stroke="transparent" legendType="none" />
              <BoxplotLayer />
            </BarChart>
          </ResponsiveContainer>
        );
      }
      case 'clustering': {
        const hasTopCategories = Boolean(config.cluster_profiles?.some((profile: any) => profile.top_categories?.length));
        const aggregationLabel = config.aggregation_label || (config.aggregation_method === 'median' ? 'Features shown as: Median values' : 'Features shown as: Average values');
        const silhouetteText = config.silhouette_text || 'Silhouette Score: Not available.';
        return (
          <div className="clustering-container">
            <ResponsiveContainer width="100%" height={300} debounce={50}>
              <ScatterChart margin={{ top: 20, right: 24, bottom: 48, left: 12 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--border-color)" />
                <XAxis
                  type="number"
                  dataKey={config.scatter_col_x}
                  name={config.scatter_col_x}
                  stroke="var(--text-muted)"
                  fontSize={12}
                  tickFormatter={yAxisFormatter}
                  label={{ value: config.xAxisLabel || config.scatter_col_x, position: 'insideBottom', offset: -30, fill: 'var(--text-muted)', fontSize: 11 }}
                />
                <YAxis
                  type="number"
                  dataKey={config.scatter_col_y}
                  name={config.scatter_col_y}
                  stroke="var(--text-muted)"
                  fontSize={12}
                  tickFormatter={yAxisFormatter}
                  label={{ value: config.yAxisLabel || config.scatter_col_y, angle: -90, position: 'insideLeft', fill: 'var(--text-muted)', fontSize: 11 }}
                />
                <RechartsTooltip cursor={{ strokeDasharray: '3 3' }} content={renderPointTooltip} />
                <Scatter name={c.clusters} data={chartDataToUse} shape={renderScatterShape} onClick={exportMode ? undefined : (point: any) => setLockedPoint(point?.payload || point)} isAnimationActive={false}>
                  {chartDataToUse.map((entry: any, index: number) => (
                    <Cell key={`cell-${index}`} fill={COLORS[parseInt(entry.cluster) % COLORS.length]} />
                  ))}
                </Scatter>
              </ScatterChart>
            </ResponsiveContainer>
            {renderLockedPointPanel(config.scatter_col_x, config.scatter_col_y)}
            <div className="cluster-meta-strip">
              <span>{aggregationLabel}</span>
              <span className={`silhouette-quality ${String(config.silhouette_quality || '').toLowerCase()}`}>{silhouetteText}</span>
            </div>
            {config.projection_note && (
              <div className="cluster-projection-note">{config.projection_note}</div>
            )}
            {config.cluster_profiles && (
              <div className="cluster-profiles-table mt-4">
                <table className="analysis-table mini">
                  <thead>
                    <tr>
                      <th>{c.clusterHeader || 'Cluster'}</th>
                      <th>{c.clusterSize}</th>
                      <th>{c.clusterFeatures}</th>
                      {hasTopCategories && <th>Dominant categories</th>}
                    </tr>
                  </thead>
                  <tbody>
                    {config.cluster_profiles.map((profile: any) => (
                      <tr key={profile.cluster_id}>
                        <td><span className="cluster-dot" style={{ backgroundColor: COLORS[profile.cluster_id % COLORS.length] }} /> {profile.cluster_name || `#${profile.cluster_id}`}</td>
                        <td>{profile.size_pct}%</td>
                        <td className="text-xs">{Object.keys(profile).filter(k => k.endsWith('_mean')).map(k => (<div key={k}>{k.replace('_mean', '')}: {yAxisFormatter(profile[k])}</div>))}</td>
                        {hasTopCategories && (
                          <td className="text-xs cluster-top-categories">
                            {profile.top_categories?.length ? profile.top_categories.map((category: any) => (
                              <div key={category.column} className="top-category-row">
                                <span>Top {String(category.label || category.column).toLowerCase()}:</span>
                                <b>{category.values?.map((item: any) => item.value).slice(0, 3).join(', ')}</b>
                              </div>
                            )) : <span className="text-muted">-</span>}
                          </td>
                        )}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        );
      }
      case 'table':
        return (
          <div className="correlation-table-wrapper">
            <table className="analysis-table">
              <thead><tr><th>{c.variableA}</th><th>{c.variableB}</th><th>{c.strength}</th></tr></thead>
              <tbody>
                {(chartDataToUse as any[]).map((row, i) => (
                  <tr key={i}>
                    <td>{row['Var A']}</td>
                    <td>{row['Var B']}</td>
                    <td className={Math.abs(row['Correlation']) > 0.6 ? 'high-corr' : ''}>{(row['Correlation'] * 100).toFixed(0)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        );
      default:
        return <div className="chart-placeholder">{c.unsupported}</div>;
    }
  };

  return (
    <div className={`chart-widget ${exportMode ? 'chart-widget--export export-card' : ''}`}>
      <div className="chart-header">
        {exportMode ? (
          <h3 className="chart-title-static">{editableTitle}</h3>
        ) : (
          <input className="chart-title-input" value={editableTitle} onChange={(e) => handleTitleChange(e.target.value)} title={c.clickToEdit || c.editTitle || 'Click to edit'} />
        )}

        {!exportMode && <div className="chart-header-actions no-export">
          {onRemove && (
            <button className="icon-btn-ghost chart-remove-btn" onClick={onRemove} title={c.remove}>
              <X size={14} />
            </button>
          )}
          <div className="card-actions-wrapper" ref={menuRef}>
            <button className="icon-btn-ghost" onClick={() => setShowMenu(!showMenu)}>
              <MoreHorizontal size={16} />
            </button>
            {showMenu && (
              <div className="card-dropdown">
                <div className="dropdown-label">{c.format}</div>
                <button className={`dropdown-item ${activeFormat === 'compact' ? 'active' : ''}`} onClick={() => { setLocalFormat('compact'); setShowMenu(false); }}>{c.compact}</button>
                <button className={`dropdown-item ${activeFormat === 'full' ? 'active' : ''}`} onClick={() => { setLocalFormat('full'); setShowMenu(false); }}>{c.full}</button>
                {canSwitch && (
                  <>
                    <div className="dropdown-divider" />
                    <div className="dropdown-label">{c.switchType}</div>
                    <button className={`dropdown-item ${activeType === 'bar' ? 'active' : ''}`} onClick={() => handleTypeSwitch('bar')}>{c.typeBar}</button>
                    <button className={`dropdown-item ${activeType === 'line' ? 'active' : ''}`} onClick={() => handleTypeSwitch('line')}>{c.typeLine}</button>
                    <button className={`dropdown-item ${activeType === 'area' ? 'active' : ''}`} onClick={() => handleTypeSwitch('area')}>{c.typeArea}</button>
                  </>
                )}
              </div>

            )}
          </div>
        </div>}
      </div>

      <div className="chart-body">
        {config.warnings && config.warnings.length > 0 && (
          <div className="chart-warnings">
            {config.warnings.map(w => (
              <div key={w} className={`warning-item ${w}`}>
                {w === 'low_r2' ? <AlertTriangle size={14} /> : <ShieldAlert size={14} />}
                <span>
                  {w === 'low_r2' && c.warningLowR2}
                  {w === 'low_data' && c.warningLowData}
                  {w === 'high_error' && c.warningHighError}
                  {w === 'low_silhouette' && c.warningLowSilhouette}
                  {w === 'high_overlap' && c.warningHighOverlap}
                  {w === 'cluster_imbalance' && c.warningClusterImbalance}
                </span>
              </div>
            ))}
          </div>
        )}
        {renderChart()}
      </div>

      {hasVisibleInsight && (
        <div className="chart-footer">
          <div className="insight-top">
            <Info size={16} className="insight-icon" />
            {exportMode ? (
              <div className="insight-editor">
                <span>{c.insight}</span>
                <p className="insight-text insight-text-static">{editableInsight}</p>
              </div>
            ) : (
              <label className="insight-editor">
                <span>{c.insight}</span>
                <textarea
                  className="insight-text insight-text-input"
                  value={editableInsight}
                  onChange={(e) => handleInsightChange(e.target.value)}
                  title={c.editInsight || c.editTitle}
                  rows={2}
                />
              </label>
            )}
          </div>
          {(config.normalityNote || config.rSquared || config.cramersV || config.effect_size !== undefined || config.quality) && (
            <div className="stats-badges">
              {config.normalityNote && <span className={`stat-badge ${config.isNormal ? 'normal' : 'skewed'}`}>{config.isNormal ? (c.normalityNormal || c.normalDistribution || config.normalityNote) : (c.normalitySkewed || c.skewedDistribution || config.normalityNote)}</span>}
              {config.rSquared && <span className="stat-badge r2">R² = {config.rSquared.toFixed(3)}</span>}
              {config.cramersV && <span className="stat-badge cv">Cramér's V = {config.cramersV.toFixed(3)}</span>}
              {config.effect_size !== undefined && <span className="stat-badge">{c.effect}: {config.effect_size.toFixed(2)} {config.effect_label ? `(${config.effect_label})` : ''}</span>}
              {config.quality && <span className="stat-badge">{c.quality}: {config.quality}</span>}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
