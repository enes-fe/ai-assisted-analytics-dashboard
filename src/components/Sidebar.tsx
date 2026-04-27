import { useState, useEffect, useRef } from 'react';
import { Database, FileText, BarChart3, RefreshCw, PlusCircle, X, Trash2, AlertTriangle } from 'lucide-react';
import { useToast } from './useToast';
import { useLang } from '../contexts/useLang';
import type { DatasetSelection } from '../types';
import './Sidebar.css';


interface DatasetMeta {
  id: number;
  filename: string;
  row_count: number;
  created_at: string;
}

interface SidebarProps {
  currentDatasetId: number | null;
  onSelect: (dataset: DatasetSelection) => void;
  onNewDataset: () => void;
}

interface ConfirmDialogState {
  open: boolean;
  title: string;
  message: string;
  onConfirm: () => void;
}

export default function Sidebar({ currentDatasetId, onSelect, onNewDataset }: SidebarProps) {
  const [datasets, setDatasets] = useState<DatasetMeta[]>([]);
  const [loading, setLoading] = useState(false);
  const [confirmDialog, setConfirmDialog] = useState<ConfirmDialogState>({
    open: false, title: '', message: '', onConfirm: () => {},
  });
  const { showToast } = useToast();
  const { t } = useLang();
  const s = t.sidebar;
  // P2: throttle — only re-fetch if 30 s have elapsed since last successful fetch
  const lastFetchRef = useRef<number>(0);
  const FETCH_INTERVAL_MS = 30_000;

  const closeDialog = () => setConfirmDialog(d => ({ ...d, open: false }));


  const fetchDatasets = async (force = false) => {
    // Throttle: skip if recent fetch exists and not forced
    if (!force && Date.now() - lastFetchRef.current < FETCH_INTERVAL_MS) return;
    setLoading(true);
    try {
      const res = await fetch('/api/datasets');
      if (res.ok) {
        const data = await res.json();
        setDatasets(data);
        lastFetchRef.current = Date.now();
      }
    } catch (e) {
      console.error('Failed to fetch datasets', e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDatasets();
  }, []);


  const handleDelete = (e: React.MouseEvent, ds: DatasetMeta) => {
    e.stopPropagation();
    setConfirmDialog({
      open: true,
      title: s.confirmDeleteTitle,
      message: s.confirmDeleteMsg(ds.filename),
      onConfirm: async () => {
        closeDialog();
        try {
          const res = await fetch(`/api/datasets/${ds.id}`, { method: 'DELETE' });
          if (res.ok) {
            setDatasets(prev => prev.filter(d => d.id !== ds.id));
            showToast(s.deleteSuccess, 'success');
          } else {
            throw new Error(`HTTP ${res.status}`);
          }
        } catch (e: any) {
          showToast(`${s.deleteFail} — ${e?.message || ''}`, 'error');
        }
      },
    });
  };

  const handleClearAll = () => {
    if (!datasets.length) return;
    setConfirmDialog({
      open: true,
      title: s.confirmClearTitle,
      message: s.confirmClearMsg,
      onConfirm: async () => {
        closeDialog();
        setLoading(true);
        try {
          const results = await Promise.allSettled(
            datasets.map(ds => fetch(`/api/datasets/${ds.id}`, { method: 'DELETE' }))
          );
          const failed = results.filter(r => r.status === 'rejected').length;
          setDatasets([]);
          if (currentDatasetId) onNewDataset();
          if (failed > 0) {
            showToast(`${s.clearAllFail} (${failed} başarısız)`, 'warning');
          } else {
            showToast(s.deleteSuccess, 'success');
          }
        } catch (e: any) {
          showToast(`${s.clearAllFail} — ${e?.message || ''}`, 'error');
        } finally {
          setLoading(false);
        }
      },
    });
  };

  const handleSelect = async (dataset: DatasetMeta) => {
    if (dataset.id === currentDatasetId) return;
    try {
      const res = await fetch(`/api/data/${dataset.id}?page=1&page_size=150`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const dataResponse = await res.json();
      onSelect({
        dataset_id: dataset.id,
        filename: dataset.filename,
        row_count: dataset.row_count,
        data: dataResponse.data,
        columns: dataResponse.data?.length > 0 ? Object.keys(dataResponse.data[0]) : [],
      });
    } catch (e: any) {
      showToast(`${s.loadError} — ${e?.message || ''}`, 'error');
    }
  };

  return (
    <aside className="sidebar">
      {/* Confirm Dialog */}
      {confirmDialog.open && (
        <div className="sidebar-confirm-overlay">
          <div className="sidebar-confirm-dialog">
            <div className="confirm-icon"><AlertTriangle size={20} /></div>
            <div className="confirm-title">{confirmDialog.title}</div>
            <div className="confirm-message">{confirmDialog.message}</div>
            <div className="confirm-actions">
              <button className="confirm-btn-cancel" onClick={closeDialog}>{s.cancel}</button>
              <button className="confirm-btn-ok" onClick={confirmDialog.onConfirm}>{s.confirm}</button>
            </div>
          </div>
        </div>
      )}

      {/* Brand */}
      <div className="sidebar-brand">
        <div className="logo-mark"><BarChart3 size={16} /></div>
        <span className="logo-text">{s.brand}</span>
      </div>

      {/* Library Header */}
      <div className="sidebar-section-header">
        <div className="sidebar-section-title">
          <Database size={14} />
          <span>{s.library}</span>
        </div>
        <div style={{ display: 'flex', gap: '4px' }}>
          <button
            className="sidebar-icon-btn"
            onClick={handleClearAll}
            disabled={loading || datasets.length === 0}
            title={s.clearAll}
          >
            <Trash2 size={13} style={{ color: datasets.length ? '#ef4444' : 'inherit' }} />
          </button>
          <button
            className="sidebar-icon-btn"
            onClick={() => fetchDatasets(true)}
            disabled={loading}
            title={s.refresh}
          >
            <RefreshCw size={13} className={loading ? 'spin' : ''} />
          </button>
        </div>
      </div>

      {/* New Dataset Button */}
      <div className="sidebar-new-btn-wrapper">
        <button className="sidebar-new-btn" onClick={onNewDataset}>
          <PlusCircle size={15} />
          <span>{s.newDataset}</span>
        </button>
      </div>

      {/* Dataset List */}
      <div className="sidebar-content">
        {loading && (
          <div className="sidebar-status">
            <div className="sidebar-spinner" />
            <span>{s.loading}</span>
          </div>
        )}

        {!loading && datasets.length === 0 && (
          <div className="sidebar-empty">
            <Database size={28} className="sidebar-empty-icon" />
            <p>{s.empty}</p>
            <p className="sidebar-empty-sub">{s.emptySub}</p>
          </div>
        )}

        {datasets.map(ds => (
          <div
            key={ds.id}
            className={`ds-item ${ds.id === currentDatasetId ? 'active' : ''}`}
            onClick={() => handleSelect(ds)}
            title={ds.filename}
          >
            <div className="ds-item-icon"><FileText size={15} /></div>
            <div className="ds-item-info">
              <div className="ds-item-name">{ds.filename}</div>
              <div className="ds-item-meta">
                <span>{ds.row_count.toLocaleString()} {s.records}</span>
                <span className="ds-dot">•</span>
                <span>{new Date(ds.created_at).toLocaleDateString('tr-TR')}</span>
              </div>
            </div>
            <button
              className="ds-delete-btn"
              onClick={e => handleDelete(e, ds)}
              title={s.delete}
            >
              <X size={13} />
            </button>
          </div>
        ))}
      </div>
    </aside>
  );
}
