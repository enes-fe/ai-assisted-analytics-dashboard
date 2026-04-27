import { useState, useRef, useEffect, useMemo, useCallback } from 'react';
import DataUploader from './DataUploader';
import AIPrompt from './AIPrompt';
import ChartWidget from './ChartWidget';
import type { ChartConfig } from './ChartWidget';
import KPICard from './KPICard';
import type { KPIConfig } from './KPICard';
import DataGrid from './DataGrid';
import { Download, Settings, FileText, LayoutGrid, Unlock } from 'lucide-react';
import { useToast } from './useToast';
import { useLang } from '../contexts/useLang';
import html2canvas from 'html2canvas';
import jsPDF from 'jspdf';
import { ResponsiveGridLayout } from 'react-grid-layout';
import type { PendingSelection, UploadResponse, DataRow } from '../types';
import 'react-grid-layout/css/styles.css';
import 'react-resizable/css/styles.css';
import './Dashboard.css';


type AnalysisPhase = 'idle' | 'uploading' | 'reading' | 'clustering' | 'metrics' | 'complete';

interface DashboardProps {
  pendingSelection?: PendingSelection | null;
  onPendingConsumed?: () => void;
  onDatasetIdChange?: (id: number | null) => void;
}

export default function Dashboard({ pendingSelection, onPendingConsumed, onDatasetIdChange }: DashboardProps) {
  const [datasetId, setDatasetId] = useState<number | null>(null);
  const [data, setData] = useState<DataRow[] | null>(null);
  const [columns, setColumns] = useState<string[]>([]);
  const [charts, setCharts] = useState<ChartConfig[]>([]);
  const [kpiData, setKpiData] = useState<KPIConfig[]>([]);
  const [rowCount, setRowCount] = useState(0);
  const [filename, setFilename] = useState('');

  const [analysisPhase, setAnalysisPhase] = useState<AnalysisPhase>('idle');
  const [isReArchitecting, setIsReArchitecting] = useState(false);
  const [activeTab, setActiveTab] = useState<'insights' | 'data'>('insights');
  const [numberFormat, setNumberFormat] = useState<'compact' | 'full'>('compact');
  const [showSettings, setShowSettings] = useState(false);
  const [mlCharts, setMlCharts] = useState<ChartConfig[]>([]);
  const [isGridLayout, setIsGridLayout] = useState(false);
  const [layouts, setLayouts] = useState<any>(() => {
    try {
      const saved = localStorage.getItem('dashboard_layouts');
      return saved ? JSON.parse(saved) : {};
    } catch {
      return {};
    }
  });

  const [containerWidth, setContainerWidth] = useState(1200);
  const containerRef = useRef<HTMLDivElement>(null);
  const layoutTimeoutRef = useRef<any>(null);
  const { showToast } = useToast();
  const { t } = useLang();
  const d = t.dashboard;
  const settingsRef = useRef<HTMLDivElement>(null);


  // ─── Resize observer ───────────────────────────────────────────
  useEffect(() => {
    const handleResize = () => {
      if (containerRef.current) setContainerWidth(containerRef.current.offsetWidth);
    };
    window.addEventListener('resize', handleResize);
    handleResize();
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // ─── Persist layouts (debounced via useEffect dependency) ──────
  useEffect(() => {
    const id = setTimeout(() => {
      localStorage.setItem('dashboard_layouts', JSON.stringify(layouts));
    }, 300);
    return () => clearTimeout(id);
  }, [layouts]);

  // ─── Close settings on outside click ──────────────────────────
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (settingsRef.current && !settingsRef.current.contains(event.target as Node)) {
        setShowSettings(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // ─── Consume pendingSelection from sidebar ─────────────────────
  useEffect(() => {
    if (!pendingSelection) return;
    onPendingConsumed?.();

    if ('reset' in pendingSelection) {
      // User clicked "Yeni Veri Seti" → go back to upload
      setDatasetId(null);
      setData(null);
      setColumns([]);
      setCharts([]);
      setKpiData([]);
      setMlCharts([]);
      setAnalysisPhase('idle');
      onDatasetIdChange?.(null);
      return;
    }

    const selection = pendingSelection;
    setDatasetId(selection.dataset_id);
    setFilename(selection.filename);
    setRowCount(selection.row_count);
    setColumns(selection.columns);
    setData(selection.data);

    setKpiData([]);
    setCharts([]);
    setMlCharts([]);
    setAnalysisPhase('reading');

    onDatasetIdChange?.(selection.dataset_id);

    setTimeout(() => {
      setAnalysisPhase('metrics');
      fetchCoreAnalytics(selection.dataset_id);
    }, 400);

    showToast(t.sidebar.uploadedToast(selection.filename), 'success');
  }, [pendingSelection]);


  // ─── Upload handlers ───────────────────────────────────────────
  const handleUploadStart = () => setAnalysisPhase('uploading');
  const handleUploadError = () => setAnalysisPhase('idle');

  const handleDataLoaded = (response: UploadResponse) => {
    setAnalysisPhase('reading');
    setDatasetId(response.dataset_id);
    setColumns(response.columns);
    setData(response.data);
    setRowCount(response.row_count);
    setFilename(response.filename || 'Yeni Veri Seti');

    setKpiData([]);
    setCharts([]);
    setMlCharts([]);

    onDatasetIdChange?.(response.dataset_id);

    setTimeout(() => {
      setAnalysisPhase('metrics');
      fetchCoreAnalytics(response.dataset_id);
    }, 500);
  };

  // ─── Analytics fetchers ────────────────────────────────────────
  const fetchCoreAnalytics = async (id: number) => {
    try {
      const res = await fetch(`/api/analytics/core/${id}`);
      if (!res.ok) throw new Error('Core analytics failed');
      const resData = await res.json();

      setKpiData(resData.kpis);
      setCharts(resData.charts);
      setAnalysisPhase('complete');
      fetchMLAnalytics(id);
    } catch (e) {
      console.error('Core Analytics failed', e);
      showToast(d.coreError, 'error');
      setAnalysisPhase('complete');
    }

  };

  const fetchMLAnalytics = async (id: number, clusterCols?: string[]) => {
    try {
      fetch(`/api/ml/forecast/${id}`)
        .then(res => res.json())
        .then(resData => {
          if (resData.charts) {
            setMlCharts(prev => [...prev.filter(c => c.type !== 'forecast'), ...resData.charts]);
          }
        })
        .catch(() => { });

      const clusterRes = await fetch(`/api/ml/cluster/${id}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ selected_cols: clusterCols || null }),
      });
      const clusterData = await clusterRes.json();
      if (clusterData.charts) {
        setMlCharts(prev => [...prev.filter(c => c.type !== 'clustering'), ...clusterData.charts]);
      }
    } catch (e) {
      console.error('ML Analytics failed', e);
    }
  };

  // ─── AI Prompt ─────────────────────────────────────────────────
  const handleAIGenerate = async (prompt: string) => {
    if (!datasetId) return;
    setIsReArchitecting(true);
    try {
      const formData = new FormData();
      formData.append('dataset_id', datasetId.toString());
      formData.append('prompt', prompt);
      const response = await fetch('/api/chat', { method: 'POST', body: formData });
      if (!response.ok) throw new Error('Chat request failed');
      const resData = await response.json();

      if (!resData.charts || resData.charts.length === 0) {
        showToast(d.noChart, 'error');
        setIsReArchitecting(false);
        return;
      }
      setTimeout(() => {
        setCharts(prev => [...prev, ...resData.charts]);
        showToast(d.newAnalysis, 'success');
        setIsReArchitecting(false);
      }, 800);
    } catch (e) {
      console.error(e);
      showToast(d.chatError, 'error');
      setIsReArchitecting(false);
    }
  };


  // ─── Export handlers ───────────────────────────────────────────
  const handleExportCSV = () => {
    if (!data || data.length === 0) return;
    try {
      const headers = Object.keys(data[0]);
      const csvContent = [
        headers.join(','),
        ...data.map(row =>
          headers.map(h => `"${String(row[h] || '').replace(/"/g, '""')}"`).join(',')
        ),
      ].join('\n');
      const blob = new Blob(['\ufeff', csvContent], { type: 'text/csv;charset=utf-8;' });
      const link = document.createElement('a');
      link.href = URL.createObjectURL(blob);
      link.download = `export_${filename || 'dataset'}.csv`;
      link.click();
      showToast(d.csvSuccess, 'success');
    } catch {
      showToast(d.csvFail, 'error');
    }
  };


  const handleExportPDF = async () => {
    const element = document.querySelector('.insights-container') as HTMLElement;
    if (!element) return;
    showToast(d.pdfPrep, 'info');

    try {
      const canvas = await html2canvas(element, {
        scale: 1.5,
        useCORS: true,
        logging: false,
        backgroundColor: getComputedStyle(document.documentElement)
          .getPropertyValue('--bg-app')
          .trim(),
        scrollX: -window.scrollX, // PDF Kenar kesilme düzeltmesi
        scrollY: -window.scrollY, // PDF Kenar kesilme düzeltmesi
        windowWidth: element.scrollWidth,
        windowHeight: element.scrollHeight,
      });

      const pdf = new jsPDF('p', 'mm', 'a4');
      const pdfWidth = pdf.internal.pageSize.getWidth();
      const pdfHeight = pdf.internal.pageSize.getHeight();
      const margin = 10;
      const usableWidth = pdfWidth - margin * 2;
      const usableHeight = pdfHeight - margin * 2;

      const imgWidthPx = canvas.width;
      const imgHeightPx = canvas.height;

      const scaledW = usableWidth;
      const scaledH = (imgHeightPx / imgWidthPx) * scaledW;

      let yOffset = 0;
      let pageCount = 0;

      while (yOffset < scaledH) {
        if (pageCount > 0) pdf.addPage();
        const srcY = (yOffset / scaledH) * imgHeightPx;
        const pageImgH = Math.min(usableHeight, scaledH - yOffset);
        const srcH = (pageImgH / scaledH) * imgHeightPx;

        const pageCanvas = document.createElement('canvas');
        pageCanvas.width = imgWidthPx;
        pageCanvas.height = srcH;
        const ctx = pageCanvas.getContext('2d')!;
        ctx.drawImage(canvas, 0, srcY, imgWidthPx, srcH, 0, 0, imgWidthPx, srcH);
        const pageData = pageCanvas.toDataURL('image/jpeg', 0.92);

        pdf.addImage(pageData, 'JPEG', margin, margin, scaledW, pageImgH);
        yOffset += usableHeight;
        pageCount++;
      }

      pdf.save(`report_${filename || 'dataset'}.pdf`);
      showToast(d.pdfSuccess, 'success');
    } catch (e) {
      console.error(e);
      showToast(d.pdfFail, 'error');
    }
  };


  // ─── Grid layout change (debounced for performance) ───────
  const onLayoutChange = useCallback((_: any, allLayouts: any) => {
    if (layoutTimeoutRef.current) clearTimeout(layoutTimeoutRef.current);
    layoutTimeoutRef.current = setTimeout(() => {
      if (isGridLayout) setLayouts(allLayouts);
    }, 300);
  }, [isGridLayout]);

  const allCharts = useMemo(() => [...charts, ...mlCharts], [charts, mlCharts]);

  // Remove a chart by id (works for both AI-generated and heuristic charts)
  const handleRemoveChart = useCallback((chartId: string) => {
    setCharts(prev => prev.filter(c => c.id !== chartId));
    setMlCharts(prev => prev.filter(c => c.id !== chartId));
    showToast(d.chartRemoved, 'info');
  }, [d.chartRemoved, showToast]);


  // ─── Loading state ─────────────────────────────────────────────
  if (analysisPhase !== 'idle' && analysisPhase !== 'complete') {
    const steps = [
      { phase: 'uploading', label: d.loading.uploading, sub: d.loading.uploadSub },
      { phase: 'reading',   label: d.loading.reading,   sub: d.loading.analysisSub },
      { phase: 'metrics',   label: d.loading.metrics,   sub: d.loading.analysisSub },
    ];
    const currentStep = steps.findIndex(s => s.phase === analysisPhase);
    const progressPct = ((currentStep + 1) / steps.length) * 100;
    const active = steps[currentStep] || steps[0];

    return (
      <div className="dashboard-container center-content">
        <div className="analysis-loader-container">
          <div className="step-progress-wrapper">
            {/* Step indicators */}
            <div className="step-indicators">
              {steps.map((s, i) => (
                <div key={s.phase} className={`step-dot-group ${i < currentStep ? 'done' : i === currentStep ? 'active' : 'pending'}`}>
                  <div className="step-dot">
                    {i < currentStep ? '✓' : i + 1}
                  </div>
                  {i < steps.length - 1 && <div className="step-line" />}
                </div>
              ))}
            </div>
            {/* Progress bar */}
            <div className="step-progress-bar">
              <div className="step-progress-fill" style={{ width: `${progressPct}%` }} />
            </div>
          </div>
          <div className="spinner" />
          <div className="analysis-text">{active.label}</div>
          <div className="analysis-subtext">{active.sub}</div>
          <button className="btn-secondary" style={{ marginTop: '2rem', background: 'rgba(239, 68, 68, 0.1)', color: '#ef4444', borderColor: 'rgba(239, 68, 68, 0.2)' }} onClick={() => setAnalysisPhase('idle')}>
            {d.loading.cancel}
          </button>
        </div>
      </div>
    );

  }


  // ─── Upload state ──────────────────────────────────────────────
  if (!datasetId && analysisPhase === 'idle') {
    return (
      <div className="dashboard-container center-content">
        <DataUploader
          onUploadStart={handleUploadStart}
          onDataLoaded={handleDataLoaded}
          onUploadError={handleUploadError}
        />
      </div>
    );
  }

  // ─── Main dashboard ────────────────────────────────────────────
  return (
    <div className="dashboard-container">
      {/* Header bar */}
      <div className="dashboard-header">
        <div className="dashboard-title-group">
          <h2 className="page-title" title={filename}>{filename || 'Analitik Panel'}</h2>
          <p className="page-subtitle">{rowCount.toLocaleString('tr-TR')} {d.subtitle}</p>
        </div>

        <div className="dashboard-center-group">
          <div className="dashboard-tabs">
            <button className={`tab-btn ${activeTab === 'insights' ? 'active' : ''}`} onClick={() => setActiveTab('insights')}>{d.tabs.insights}</button>
            <button className={`tab-btn ${activeTab === 'data' ? 'active' : ''}`} onClick={() => setActiveTab('data')}>{d.tabs.data}</button>
          </div>
        </div>

        <div className="dashboard-actions">
          <div className="settings-wrapper" ref={settingsRef}>
            <button className={`btn-secondary icon-btn-only ${showSettings ? 'active' : ''}`} onClick={() => setShowSettings(!showSettings)} title="Ayarlar">
              <Settings size={18} />
            </button>
            {showSettings && (
              <div className="card-dropdown global-settings">
                <div className="dropdown-label">{d.settings.numberFormat}</div>
                <button className={`dropdown-item ${numberFormat === 'compact' ? 'active' : ''}`} onClick={() => { setNumberFormat('compact'); setShowSettings(false); }}>{d.settings.compact}</button>
                <button className={`dropdown-item ${numberFormat === 'full' ? 'active' : ''}`} onClick={() => { setNumberFormat('full'); setShowSettings(false); }}>{d.settings.full}</button>
              </div>
            )}
          </div>

          <button className={`btn-secondary ${isGridLayout ? 'active' : ''}`} onClick={() => setIsGridLayout(!isGridLayout)} title={isGridLayout ? d.actions.editMode : d.actions.dynamicLayout}>
            {isGridLayout ? <Unlock size={16} /> : <LayoutGrid size={16} />}
            <span>{isGridLayout ? d.actions.editMode : d.actions.dynamicLayout}</span>
          </button>

          <button className="btn-secondary" onClick={handleExportPDF} title={d.actions.exportPdf}>
            <FileText size={16} /> {d.actions.exportPdf}
          </button>
          <button className="btn-primary" onClick={handleExportCSV} title={d.actions.exportCsv}>
            <Download size={16} /> {d.actions.exportCsv}
          </button>
        </div>
      </div>


      {/* Tab content */}
      {activeTab === 'data' ? (
        <DataGrid datasetId={datasetId!} columns={columns} />
      ) : (
        <div className="insights-container">
          <AIPrompt onGenerate={handleAIGenerate} columns={columns} />

          {isReArchitecting ? (
            <div className="analysis-loader-container" style={{ height: '40vh' }}>
              <div className="spinner" />
              <div className="analysis-text">{d.loading.queryAnalyzing}</div>
            </div>
          ) : (
            <>
              {kpiData.length > 0 && (
                <div className="kpi-grid">
                  {kpiData.map(kpi => <KPICard key={kpi.id} config={kpi} numberFormat={numberFormat} />)}
                </div>
              )}

              {isGridLayout ? (
                <div ref={containerRef} className="charts-grid-container">
                  <ResponsiveGridLayout
                    className="charts-grid-layout"
                    layouts={layouts}
                    breakpoints={{ lg: 1200, md: 996, sm: 768, xs: 480, xxs: 0 }}
                    cols={{ lg: 12, md: 10, sm: 6, xs: 4, xxs: 2 }}
                    rowHeight={150}
                    width={containerWidth}
                    onLayoutChange={onLayoutChange}
                    dragConfig={{ handle: '.chart-header' }}
                  >
                    {allCharts.map((chart, idx) => {
                      const isWide = chart.type === 'table' || chart.type === 'clustering' || chart.type === 'forecast';
                      const savedPos = layouts?.lg?.find((l: any) => l.i === chart.id);
                      const col = idx % 2;
                      const row = Math.floor(idx / 2);
                      return (
                        <div key={chart.id} data-grid={savedPos ? undefined : { x: isWide ? 0 : col * 6, y: row * 3, w: isWide ? 12 : 6, h: 3, minW: 4, minH: 3 }}>
                          <ChartWidget config={chart} data={data!} numberFormat={numberFormat} onRemove={() => handleRemoveChart(chart.id)} />
                        </div>
                      );
                    })}
                  </ResponsiveGridLayout>
                </div>
              ) : (
                <div className="charts-css-grid">
                  {allCharts.map(chart => (
                    <div key={chart.id} className={`chart-cell ${chart.type === 'table' || chart.type === 'clustering' || chart.type === 'forecast' ? 'chart-cell--wide' : ''}`}>
                      <ChartWidget config={chart} data={data!} numberFormat={numberFormat} onRemove={() => handleRemoveChart(chart.id)} />
                    </div>
                  ))}
                </div>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}
