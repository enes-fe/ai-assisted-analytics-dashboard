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
  type: 'bar' | 'line' | 'pie' | 'area' | 'scatter' | 'table' | 'forecast' | 'clustering' | 'boxplot';
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

export default function ChartWidget({ config, data, numberFormat = 'compact', onRemove }: ChartWidgetProps) {
  const chartDataToUse = config.chartData || data;
  const storageKey = `chart_title_${config.id}`;
  const typeKey = `chart_type_${config.id}`;

  // Chart type that can be overridden by the user (persisted)
  const SWITCHABLE: Array<ChartConfig['type']> = ['bar', 'line', 'area'];
  const canSwitch = SWITCHABLE.includes(config.type as any);

  const [editableTitle, setEditableTitle] = useState(
    () => localStorage.getItem(storageKey) || config.title
  );
  const [activeType, setActiveType] = useState<ChartConfig['type']>(
    () => (localStorage.getItem(typeKey) as ChartConfig['type']) || config.type
  );
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

  const handleTypeSwitch = (type: ChartConfig['type']) => {
    setActiveType(type);
    localStorage.setItem(typeKey, type);
    setShowMenu(false);
  };

  useEffect(() => { setLocalFormat(null); }, [numberFormat]);



  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) setShowMenu(false);
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const activeFormat = localFormat || numberFormat;

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

  const renderChart = () => {
    switch (activeType) {

      case 'bar': {
        const isVertical = config.layout === 'vertical';
        return (
          <ResponsiveContainer width="100%" height={isVertical ? 400 : 300} debounce={50}>
            <BarChart data={chartDataToUse} layout={isVertical ? 'vertical' : 'horizontal'} margin={{ top: 10, right: 30, left: isVertical ? 40 : -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--border-color)" />
              {isVertical ? (
                <>
                  <XAxis type="number" stroke="var(--text-muted)" fontSize={12} tickLine={false} axisLine={false} tickFormatter={yAxisFormatter} domain={config.isStacked ? [0, 100] : [0, 'auto']} />
                  <YAxis type="category" dataKey={config.xAxisKey} stroke="var(--text-muted)" fontSize={11} tickLine={false} axisLine={false} width={100} tickFormatter={(val) => String(val).length > 20 ? String(val).substring(0, 18) + '...' : val} />
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
                  {!config.isStacked && config.series.length === 1 && chartDataToUse.map((_: any, index: number) => (
                    <Cell key={`cell-${index}`} fill={VIBRANT_GRADIENT[index % VIBRANT_GRADIENT.length]} />
                  ))}
                </Bar>
              ))}
              {config.id.startsWith('hist-') && (
                <>
                  {config.isNormal
                    ? <ReferenceLine x={config.mean} stroke="#10b981" strokeDasharray="3 3" label={{ value: 'Mean', position: 'top', fill: '#10b981', fontSize: 10 }} />
                    : <ReferenceLine x={config.median} stroke="#f59e0b" strokeDasharray="3 3" label={{ value: 'Median', position: 'top', fill: '#f59e0b', fontSize: 10 }} />
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
      case 'pie': {
        const pieDataKey = config.series[0]?.key;
        return (
          <ResponsiveContainer width="100%" height={300} debounce={50}>
            <PieChart>
              <RechartsTooltip formatter={tooltipFormatter} contentStyle={{ backgroundColor: 'var(--bg-surface)', borderColor: 'var(--border-color)', color: 'var(--text-primary)', borderRadius: '8px', boxShadow: 'var(--shadow-md)' }} />
              <Legend wrapperStyle={{ fontSize: '12px' }} />
              <Pie data={chartDataToUse} dataKey={pieDataKey} nameKey={config.xAxisKey} cx="50%" cy="50%" outerRadius={100} fill="#8884d8" label isAnimationActive={false}>
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
          <ResponsiveContainer width="100%" height={300} debounce={50}>
            <ComposedChart margin={{ top: 20, right: 20, bottom: 20, left: -20 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--border-color)" />
              <XAxis type="number" dataKey={scatterXKey} name={scatterXKey} stroke="var(--text-muted)" fontSize={12} tickFormatter={yAxisFormatter} domain={['auto', 'auto']} />
              <YAxis type="number" dataKey={scatterYKey} name={scatterYKey} stroke="var(--text-muted)" fontSize={12} tickFormatter={yAxisFormatter} domain={['auto', 'auto']} />
              <RechartsTooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ backgroundColor: 'var(--bg-surface)', borderColor: 'var(--border-color)', borderRadius: '8px' }} />
              <Scatter name={config.title} data={chartDataToUse} fill="var(--accent)" shape={config.isHexbin ? <Hexagon /> : 'circle'} isAnimationActive={false} />
              {regressionData && (
                <Line data={regressionData} type="monotone" dataKey="regression" stroke="#ef4444" strokeWidth={2} dot={false} activeDot={false} legendType="none" isAnimationActive={false} />
              )}
            </ComposedChart>
          </ResponsiveContainer>
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
                <Line type="monotone" dataKey="actual" name="Actual" stroke="#2563eb" strokeWidth={3} dot={{ r: 4 }} isAnimationActive={false} />
                <Line type="monotone" dataKey="fitted" name="Model Fit" stroke="#93c5fd" strokeWidth={2} strokeDasharray="5 5" dot={false} isAnimationActive={false} />
                <Line type="monotone" dataKey="forecast" name="Forecast" stroke="#10b981" strokeWidth={3} strokeDasharray="5 5" dot={{ r: 5 }} isAnimationActive={false} />
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
                      <div>Max: <b>{yAxisFormatter(d.max)}</b></div>
                      <div>Q3: <b>{yAxisFormatter(d.q3)}</b></div>
                      <div>{c.median}: <b>{yAxisFormatter(d.median)}</b></div>
                      <div>Q1: <b>{yAxisFormatter(d.q1)}</b></div>
                      <div>Min: <b>{yAxisFormatter(d.min)}</b></div>
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
      case 'clustering':
        return (
          <div className="clustering-container">
            <ResponsiveContainer width="100%" height={300} debounce={50}>
              <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: -20 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--border-color)" />
                <XAxis type="number" dataKey={config.scatter_col_x} name={config.scatter_col_x} stroke="var(--text-muted)" fontSize={12} tickFormatter={yAxisFormatter} />
                <YAxis type="number" dataKey={config.scatter_col_y} name={config.scatter_col_y} stroke="var(--text-muted)" fontSize={12} tickFormatter={yAxisFormatter} />
                <RechartsTooltip cursor={{ strokeDasharray: '3 3' }} />
                <Scatter name="Clusters" data={chartDataToUse}>
                  {chartDataToUse.map((entry: any, index: number) => (
                    <Cell key={`cell-${index}`} fill={COLORS[parseInt(entry.cluster) % COLORS.length]} />
                  ))}
                </Scatter>
              </ScatterChart>
            </ResponsiveContainer>
            {config.cluster_profiles && (
              <div className="cluster-profiles-table mt-4">
                <table className="analysis-table mini">
                  <thead><tr><th>Cluster</th><th>{c.clusterSize}</th><th>{c.clusterFeatures}</th></tr></thead>
                  <tbody>
                    {config.cluster_profiles.map((profile: any) => (
                      <tr key={profile.cluster_id}>
                        <td><span className="cluster-dot" style={{ backgroundColor: COLORS[profile.cluster_id % COLORS.length] }} /> #{profile.cluster_id}</td>
                        <td>{profile.size_pct}%</td>
                        <td className="text-xs">{Object.keys(profile).filter(k => k.endsWith('_mean')).map(k => (<div key={k}>{k.replace('_mean', '')}: {yAxisFormatter(profile[k])}</div>))}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        );
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
    <div className="chart-widget">
      <div className="chart-header">
        <input className="chart-title-input" value={editableTitle} onChange={(e) => handleTitleChange(e.target.value)} title="Click to edit" />

        <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
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
        </div>
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

      {config.insight && (
        <div className="chart-footer">
          <div className="insight-top">
            <Info size={16} className="insight-icon" />
            <p className="insight-text"><strong>{c.insight}</strong> {config.insight}</p>
          </div>
          {(config.normalityNote || config.rSquared || config.cramersV || config.effect_size !== undefined || config.quality) && (
            <div className="stats-badges">
              {config.normalityNote && <span className={`stat-badge ${config.isNormal ? 'normal' : 'skewed'}`}>{config.normalityNote}</span>}
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
