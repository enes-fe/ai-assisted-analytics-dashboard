import { useState, useEffect } from 'react';
import { ArrowDown, ArrowUp, ChevronsUpDown, ChevronLeft, ChevronRight, Loader2 } from 'lucide-react';
import { useLang } from '../contexts/useLang';
import type { DataRow } from '../types';
import './DataGrid.css';

interface DataGridProps {
  datasetId: number;
  columns: string[];
}

type SortDir = 'asc' | 'desc';

export default function DataGrid({ datasetId, columns }: DataGridProps) {
  const [data, setData] = useState<DataRow[]>([]);
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [totalRows, setTotalRows] = useState(0);
  const [loading, setLoading] = useState(false);
  const [sortBy, setSortBy] = useState<string | null>(null);
  const [sortDir, setSortDir] = useState<SortDir>('asc');
  const pageSize = 50;
  const [editableTitle, setEditableTitle] = useState('');
  const [pageInput, setPageInput] = useState('1');
  const { t } = useLang();
  const g = t.grid;

  useEffect(() => { setEditableTitle(g.title); }, [g.title]);

  useEffect(() => {
    setPageInput(page.toString());
  }, [page]);

  useEffect(() => {
    setPage(1);
    setSortBy(null);
    setSortDir('asc');
  }, [datasetId]);

  const handlePageSubmit = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      let p = parseInt(pageInput);
      if (isNaN(p) || p < 1) p = 1;
      if (p > totalPages) p = totalPages;
      setPage(p);
      setPageInput(p.toString());
    }
  };

  const handleSort = (col: string) => {
    if (sortBy !== col) {
      setSortBy(col);
      setSortDir('asc');
    } else if (sortDir === 'asc') {
      setSortDir('desc');
    } else {
      setSortBy(null);
      setSortDir('asc');
    }
    setPage(1);
  };

  useEffect(() => {
    async function fetchData() {
      setLoading(true);
      try {
        const params = new URLSearchParams({
          page: page.toString(),
          page_size: pageSize.toString(),
        });
        if (sortBy) {
          params.set('sort_by', sortBy);
          params.set('sort_dir', sortDir);
        }
        const response = await fetch(`/api/data/${datasetId}?${params.toString()}`);
        if (!response.ok) throw new Error('Failed to fetch data');
        const resData = await response.json();
        setData(resData.data);
        setTotalPages(resData.total_pages);
        setTotalRows(resData.total_rows);
      } catch (err) {
        console.error(err);
      } finally {
        setLoading(false);
      }
    }
    fetchData();
  }, [datasetId, page, sortBy, sortDir]);

  const sortTitle = (col: string) => {
    if (sortBy !== col) return g.sortColumn || 'Kolona göre sırala';
    return sortDir === 'asc' ? (g.sortAsc || 'Artan sıralı') : (g.sortDesc || 'Azalan sıralı');
  };

  return (
    <div className="datagrid-widget">
      <div className="datagrid-header">
        <div className="datagrid-title">
          <input
            className="datagrid-title-input"
            value={editableTitle}
            onChange={(e) => setEditableTitle(e.target.value)}
            title={g.editTitle}
          />
          <p>{g.displaying(page, totalPages, totalRows)}</p>
        </div>

        <div className="datagrid-pagination">
          <button
            className="btn-icon"
            disabled={page === 1 || loading}
            onClick={() => setPage(p => p - 1)}
          >
            <ChevronLeft size={16} />
          </button>
          <div className="page-indicator">
            <input
              type="number"
              className="page-input"
              value={pageInput}
              onChange={(e) => setPageInput(e.target.value)}
              onKeyDown={handlePageSubmit}
              title={g.pageInput}
              min={1}
              max={totalPages}
            />
            <span> / {totalPages}</span>
          </div>
          <button
            className="btn-icon"
            disabled={page === totalPages || loading}
            onClick={() => setPage(p => p + 1)}
          >
            <ChevronRight size={16} />
          </button>
        </div>
      </div>

      <div className="datagrid-table-container">
        {loading ? (
          <div className="datagrid-loading">
            <Loader2 className="spinner-icon" size={32} />
          </div>
        ) : (
          <table className="datagrid-table">
            <thead>
              <tr>
                <th className="row-num-col">#</th>
                {columns.map(col => (
                  <th key={col}>
                    <button
                      type="button"
                      className={`datagrid-sort-btn ${sortBy === col ? 'active' : ''}`}
                      onClick={() => handleSort(col)}
                      title={sortTitle(col)}
                    >
                      <span>{col}</span>
                      {sortBy === col ? (
                        sortDir === 'asc' ? <ArrowUp size={13} /> : <ArrowDown size={13} />
                      ) : (
                        <ChevronsUpDown size={13} />
                      )}
                    </button>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data.map((row, index) => (
                <tr key={index}>
                  <td className="row-num-col">{(page - 1) * pageSize + index + 1}</td>
                  {columns.map(col => (
                    <td key={col}>{row[col] !== null ? String(row[col]) : ''}</td>
                  ))}
                </tr>
              ))}
              {data.length === 0 && (
                <tr>
                  <td colSpan={columns.length + 1} className="empty-state">{g.noData}</td>
                </tr>
              )}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
