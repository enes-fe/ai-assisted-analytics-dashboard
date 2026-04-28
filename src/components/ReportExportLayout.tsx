import ChartWidget from './ChartWidget';
import type { ChartConfig } from './ChartWidget';
import type { KPIConfig } from './KPICard';
import type { DataRow } from '../types';
import { useLang } from '../contexts/useLang';
import './ReportExportLayout.css';

type AiDashboardStatus = 'idle' | 'loading' | 'success' | 'timeout' | 'error';

interface ReportExportLayoutProps {
  datasetName: string;
  generatedAt: Date;
  rowCount: number;
  data: DataRow[];
  kpis: KPIConfig[];
  charts: ChartConfig[];
  mlCharts: ChartConfig[];
  numberFormat: 'compact' | 'full';
  aiStatus: AiDashboardStatus;
  aiMessage?: string;
  forecastMessage?: string;
}

const REPORT_COPY = {
  tr: {
    title: 'Analitik Rapor',
    dataset: 'Veri seti',
    generated: 'Uretildi',
    rows: 'Kayit sayisi',
    kpis: 'KPI Ozeti',
    mainCharts: 'Ana Grafikler',
    forecast: 'Tahmin Analizi',
    segmentation: 'Segmentasyon Analizi',
    insights: 'Icgozu Ozetleri',
    notes: 'Notlar ve Uyarilar',
    noKpis: 'Bu raporda KPI bulunmuyor.',
    noCharts: 'Bu bolum icin grafik bulunmuyor.',
    noInsights: 'Ozetlenecek icgozu bulunmuyor.',
    noNotes: 'Belirgin not veya uyari yok.',
    kpiLabel: 'KPI',
    chartLabel: 'Grafik',
    forecastLabel: 'Tahmin',
    segmentLabel: 'Segmentasyon',
    aiLoading: 'AI analizi rapor alindigi anda hazirlaniyordu.',
    aiTimeout: 'AI analizi zaman asimina ugradi.',
    aiError: 'AI analizi tamamlanamadi.',
  },
  en: {
    title: 'Analytics Report',
    dataset: 'Dataset',
    generated: 'Generated',
    rows: 'Rows',
    kpis: 'KPI Summary',
    mainCharts: 'Main Charts',
    forecast: 'Forecast Analysis',
    segmentation: 'Segmentation Analysis',
    insights: 'Insight Summaries',
    notes: 'Notes and Warnings',
    noKpis: 'No KPIs are available for this report.',
    noCharts: 'No charts are available for this section.',
    noInsights: 'No insights are available to summarize.',
    noNotes: 'No notable warnings.',
    kpiLabel: 'KPI',
    chartLabel: 'Chart',
    forecastLabel: 'Forecast',
    segmentLabel: 'Segmentation',
    aiLoading: 'AI analysis was still preparing when the report was exported.',
    aiTimeout: 'AI analysis timed out.',
    aiError: 'AI analysis could not be completed.',
  },
} as const;

function readStoredText(key: string, fallback = '') {
  if (typeof window === 'undefined') return fallback;
  try {
    const stored = window.localStorage.getItem(key);
    return stored !== null ? stored : fallback;
  } catch {
    return fallback;
  }
}

function chunkItems<T>(items: T[], size: number) {
  const chunks: T[][] = [];
  for (let index = 0; index < items.length; index += size) {
    chunks.push(items.slice(index, index + size));
  }
  return chunks;
}

export default function ReportExportLayout({
  datasetName,
  generatedAt,
  rowCount,
  data,
  kpis,
  charts,
  mlCharts,
  numberFormat,
  aiStatus,
  aiMessage,
  forecastMessage,
}: ReportExportLayoutProps) {
  const { lang, t } = useLang();
  const copy = REPORT_COPY[lang];
  const locale = lang === 'tr' ? 'tr-TR' : 'en-US';
  const allCharts = [...charts, ...mlCharts];
  const mainCharts = allCharts.filter(chart => chart.type !== 'forecast' && chart.type !== 'clustering');
  const forecastCharts = allCharts.filter(chart => chart.type === 'forecast');
  const segmentationCharts = allCharts.filter(chart => chart.type === 'clustering');

  const formatValue = (kpi: KPIConfig) => {
    if (typeof kpi.rawValue === 'number') {
      return new Intl.NumberFormat(locale, numberFormat === 'compact'
        ? { notation: 'compact', maximumFractionDigits: 2 }
        : { maximumFractionDigits: 2 }
      ).format(kpi.rawValue);
    }
    return String(kpi.value);
  };

  const warningText = (code: string) => {
    switch (code) {
      case 'low_r2':
        return t.chart.warningLowR2;
      case 'low_data':
        return t.chart.warningLowData;
      case 'high_error':
        return t.chart.warningHighError;
      case 'low_silhouette':
        return t.chart.warningLowSilhouette;
      case 'high_overlap':
        return t.chart.warningHighOverlap;
      case 'cluster_imbalance':
        return t.chart.warningClusterImbalance;
      default:
        return code.replace(/_/g, ' ');
    }
  };

  const insightItems = [
    ...kpis
      .map(kpi => ({
        id: `kpi-${kpi.id}`,
        label: copy.kpiLabel,
        title: readStoredText(`kpi_title_${kpi.id}`, kpi.title),
        text: readStoredText(`kpi_insight_${kpi.id}`, kpi.insight || ''),
      }))
      .filter(item => item.text.trim()),
    ...allCharts
      .map(chart => ({
        id: `chart-${chart.id}`,
        label: chart.type === 'forecast' ? copy.forecastLabel : chart.type === 'clustering' ? copy.segmentLabel : copy.chartLabel,
        title: readStoredText(`chart_title_${chart.id}`, chart.title),
        text: readStoredText(`chart_insight_${chart.id}`, chart.insight || ''),
      }))
      .filter(item => item.text.trim()),
  ];

  const notes = [
    ...(aiStatus === 'loading' ? [copy.aiLoading] : []),
    ...(aiStatus === 'timeout' ? [aiMessage || copy.aiTimeout] : []),
    ...(aiStatus === 'error' ? [aiMessage || copy.aiError] : []),
    ...(forecastMessage ? [forecastMessage] : []),
    ...allCharts.flatMap(chart => (chart.warnings || []).map(warning => `${readStoredText(`chart_title_${chart.id}`, chart.title)}: ${warningText(warning)}`)),
  ];

  const renderChartSections = (sectionTitle: string, sectionCharts: ChartConfig[], emptyText: string) => {
    if (sectionCharts.length === 0) {
      return (
        <section className="report-section export-section" data-export-section="true">
          <SectionHeading title={sectionTitle} />
          <p className="report-empty-state">{emptyText}</p>
        </section>
      );
    }

    return sectionCharts.map((chart, index) => (
      <section
        key={chart.id}
        className={`report-section report-chart-section report-chart-section--${chart.type} export-section`}
        data-export-section="true"
      >
        <SectionHeading title={sectionTitle} eyebrow={index > 0 ? sectionTitle : undefined} />
        <ChartWidget key={`${chart.id}-${generatedAt.getTime()}`} config={chart} data={data} numberFormat={numberFormat} exportMode />
      </section>
    ));
  };

  return (
    <article className="report-export-layout">
      <section className="report-cover export-section" data-export-section="true">
        <div>
          <p className="report-kicker">{copy.dataset}</p>
          <h1>{copy.title}</h1>
          <h2>{datasetName || 'Dataset'}</h2>
        </div>
        <dl className="report-metadata">
          <div>
            <dt>{copy.generated}</dt>
            <dd>{generatedAt.toLocaleString(locale)}</dd>
          </div>
          <div>
            <dt>{copy.rows}</dt>
            <dd>{rowCount > 0 ? rowCount.toLocaleString(locale) : '-'}</dd>
          </div>
        </dl>
      </section>

      <section className="report-section export-section" data-export-section="true">
        <SectionHeading title={copy.kpis} />
        {kpis.length > 0 ? (
          <div className="report-kpi-grid">
            {kpis.map(kpi => {
              const title = readStoredText(`kpi_title_${kpi.id}`, kpi.title);
              const insight = readStoredText(`kpi_insight_${kpi.id}`, kpi.insight || '');
              return (
                <div className="report-kpi-card export-card" key={kpi.id}>
                  <div className="report-kpi-title">{title}</div>
                  <div className="report-kpi-value">{formatValue(kpi)}</div>
                  {kpi.trend && kpi.trendDirection !== 'neutral' && (
                    <div className={`report-kpi-trend report-kpi-trend--${kpi.trendDirection}`}>{kpi.trend}</div>
                  )}
                  {insight.trim() && <p className="report-kpi-insight">{insight}</p>}
                </div>
              );
            })}
          </div>
        ) : (
          <p className="report-empty-state">{copy.noKpis}</p>
        )}
      </section>

      {renderChartSections(copy.mainCharts, mainCharts, copy.noCharts)}
      {renderChartSections(copy.forecast, forecastCharts, copy.noCharts)}
      {renderChartSections(copy.segmentation, segmentationCharts, copy.noCharts)}

      {chunkItems(insightItems, 6).map((chunk, index) => (
        <section className="report-section export-section" data-export-section="true" key={`insights-${index}`}>
          <SectionHeading title={copy.insights} eyebrow={index > 0 ? copy.insights : undefined} />
          {chunk.length > 0 ? (
            <div className="report-insight-list">
              {chunk.map(item => (
                <div className="report-insight-item export-card" key={item.id}>
                  <div className="report-insight-meta">{item.label}</div>
                  <h3>{item.title}</h3>
                  <p>{item.text}</p>
                </div>
              ))}
            </div>
          ) : (
            <p className="report-empty-state">{copy.noInsights}</p>
          )}
        </section>
      ))}

      {insightItems.length === 0 && (
        <section className="report-section export-section" data-export-section="true">
          <SectionHeading title={copy.insights} />
          <p className="report-empty-state">{copy.noInsights}</p>
        </section>
      )}

      <section className="report-section report-notes-section export-section" data-export-section="true">
        <SectionHeading title={copy.notes} />
        {notes.length > 0 ? (
          <ul className="report-notes-list">
            {notes.map((note, index) => <li key={`${note}-${index}`}>{note}</li>)}
          </ul>
        ) : (
          <p className="report-empty-state">{copy.noNotes}</p>
        )}
      </section>
    </article>
  );
}

function SectionHeading({ title, eyebrow }: { title: string; eyebrow?: string }) {
  return (
    <div className="report-section-heading">
      {eyebrow && <span>{eyebrow}</span>}
      <h2>{title}</h2>
    </div>
  );
}
