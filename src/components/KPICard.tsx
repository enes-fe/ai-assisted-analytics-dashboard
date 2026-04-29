import { useState, useEffect, useRef } from 'react';
import { ArrowDownRight, ArrowUpRight, AlertCircle, MoreHorizontal, SlidersHorizontal } from 'lucide-react';
import { useLang } from '../contexts/useLang';
import { formatNumber } from '../utils/numberFormat';
import './KPICard.css';


export interface KPIConfig {
  id: string;
  title: string;
  value: string | number;
  trend?: string;
  trendDirection?: 'up' | 'down' | 'neutral';
  insight?: string;
  rawValue?: number;
}

interface KPICardProps {
  config: KPIConfig;
  numberFormat?: 'compact' | 'full';
  onConfigure?: () => void;
}

export default function KPICard({ config, numberFormat = 'compact', onConfigure }: KPICardProps) {
  const { title, value, trend, trendDirection, insight } = config;
  const storageKey = `kpi_title_${config.id}`;
  const insightStorageKey = `kpi_insight_${config.id}`;
  const [editableTitle, setEditableTitle] = useState(
    () => localStorage.getItem(storageKey) || title
  );
  const [editableInsight, setEditableInsight] = useState(() => {
    const savedInsight = localStorage.getItem(insightStorageKey);
    return savedInsight !== null ? savedInsight : insight || '';
  });
  const [localFormat, setLocalFormat] = useState<'compact' | 'full' | null>(null);
  const [showMenu, setShowMenu] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);
  const { t, lang } = useLang();
  const k = t.kpi;
  const locale = lang === 'tr' ? 'tr-TR' : 'en-US';

  const handleTitleChange = (val: string) => {
    setEditableTitle(val);
    localStorage.setItem(storageKey, val);
  };

  const handleInsightChange = (val: string) => {
    setEditableInsight(val);
    localStorage.setItem(insightStorageKey, val);
  };

  // When global format changes, reset local override
  useEffect(() => {
    setLocalFormat(null);
  }, [numberFormat]);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        setShowMenu(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const activeFormat = localFormat || numberFormat;
  const hasVisibleInsight = editableInsight.trim().length > 0;

  const displayValue = config.rawValue !== undefined
    ? formatNumber(config.rawValue, { mode: activeFormat, locale, maximumFractionDigits: 2 })
    : formatNumber(value, { mode: activeFormat, locale, maximumFractionDigits: 2, fallback: String(value ?? '-') });

  return (
    <div className="kpi-card">
      <div className="kpi-header">
        <input
          className="kpi-title-input"
          value={editableTitle}
          onChange={(e) => handleTitleChange(e.target.value)}
          title={t.kpi.editTitle}
        />

        <div className="card-actions-wrapper no-export" ref={menuRef}>
          <button className="icon-btn-ghost" onClick={() => setShowMenu(!showMenu)}>
            <MoreHorizontal size={16} />
          </button>

          {showMenu && (
            <div className="card-dropdown">
              <div className="dropdown-label">{k.format}</div>
              <button className={`dropdown-item ${activeFormat === 'compact' ? 'active' : ''}`} onClick={() => { setLocalFormat('compact'); setShowMenu(false); }}>{k.compact}</button>
              <button className={`dropdown-item ${activeFormat === 'full' ? 'active' : ''}`} onClick={() => { setLocalFormat('full'); setShowMenu(false); }}>{k.full}</button>
              {onConfigure && (
                <>
                  <div className="dropdown-divider" />
                  <button className="dropdown-item dropdown-item-icon" onClick={() => { onConfigure(); setShowMenu(false); }}>
                    <SlidersHorizontal size={13} />
                    {k.configureColumns || 'KPI kolonlarını yönet'}
                  </button>
                </>
              )}
            </div>

          )}
        </div>
      </div>
      <div className="kpi-body">
        <div className="kpi-value">{displayValue}</div>
        {/* Plandaki gibi Nötr durum gizleniyor, sadece yukarı/aşağı trendler gösteriliyor */}
        {trend && trendDirection !== 'neutral' && (
          <div className={`kpi-trend trend-${trendDirection}`}>
            {trendDirection === 'up' && <ArrowUpRight size={14} />}
            {trendDirection === 'down' && <ArrowDownRight size={14} />}
            <span>{trend}</span>
          </div>
        )}
      </div>
      {hasVisibleInsight && (
        <div className="kpi-insight">
          <AlertCircle size={14} className="kpi-insight-icon" />
          <textarea
            className="kpi-insight-input"
            value={editableInsight}
            onChange={(e) => handleInsightChange(e.target.value)}
            title={k.editInsight || k.editTitle}
            rows={2}
          />
        </div>
      )}
    </div>
  );
}
