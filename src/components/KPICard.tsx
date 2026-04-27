import { useState, useEffect, useRef } from 'react';
import { ArrowDownRight, ArrowUpRight, AlertCircle, MoreHorizontal } from 'lucide-react';
import { useLang } from '../contexts/useLang';
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
}

export default function KPICard({ config, numberFormat = 'compact' }: KPICardProps) {
  const { title, value, trend, trendDirection, insight } = config;
  const storageKey = `kpi_title_${config.id}`;
  const [editableTitle, setEditableTitle] = useState(
    () => localStorage.getItem(storageKey) || title
  );
  const [localFormat, setLocalFormat] = useState<'compact' | 'full' | null>(null);
  const [showMenu, setShowMenu] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);
  const { t } = useLang();
  const k = t.kpi;

  const handleTitleChange = (val: string) => {
    setEditableTitle(val);
    localStorage.setItem(storageKey, val);
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

  const formatValue = (val: number, format: 'compact' | 'full') => {
    if (format === 'compact') {
      if (val >= 1_000_000_000) return `${(val / 1_000_000_000).toFixed(2)}B`;
      if (val >= 1_000_000) return `${(val / 1_000_000).toFixed(2)}M`;
      if (val >= 1_000) return `${(val / 1_000).toFixed(1)}K`;
    }
    return val % 1 !== 0 ? val.toLocaleString(undefined, { maximumFractionDigits: 2 }) : val.toLocaleString();
  };

  const displayValue = config.rawValue !== undefined ? formatValue(config.rawValue, activeFormat) : value;

  return (
    <div className="kpi-card">
      <div className="kpi-header">
        <input
          className="kpi-title-input"
          value={editableTitle}
          onChange={(e) => handleTitleChange(e.target.value)}
          title="Click to edit"
        />

        <div className="card-actions-wrapper" ref={menuRef}>
          <button className="icon-btn-ghost" onClick={() => setShowMenu(!showMenu)}>
            <MoreHorizontal size={16} />
          </button>

          {showMenu && (
            <div className="card-dropdown">
              <div className="dropdown-label">{k.format}</div>
              <button className={`dropdown-item ${activeFormat === 'compact' ? 'active' : ''}`} onClick={() => { setLocalFormat('compact'); setShowMenu(false); }}>{k.compact}</button>
              <button className={`dropdown-item ${activeFormat === 'full' ? 'active' : ''}`} onClick={() => { setLocalFormat('full'); setShowMenu(false); }}>{k.full}</button>
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
      {insight && (
        <div className="kpi-insight">
          <AlertCircle size={14} className="kpi-insight-icon" />
          <span>{insight}</span>
        </div>
      )}
    </div>
  );
}
