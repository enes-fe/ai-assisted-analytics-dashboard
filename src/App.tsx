import { useState, useEffect } from 'react';
import Header from './components/Header';
import Dashboard from './components/Dashboard';
import Sidebar from './components/Sidebar';
import type { PendingSelection } from './types';

function App() {
  const [theme, setTheme] = useState<'light' | 'dark'>(() => {
    const saved = localStorage.getItem('theme');
    return (saved as 'light' | 'dark') || 'dark';
  });
  const [sidebarOpen, setSidebarOpen] = useState(true);

  // Dataset state lifted to App so Sidebar and Dashboard can share it
  const [currentDatasetId, setCurrentDatasetId] = useState<number | null>(null);
  const [pendingSelection, setPendingSelection] = useState<PendingSelection | null>(null);

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme(t => (t === 'light' ? 'dark' : 'light'));
  };

  const toggleSidebar = () => {
    setSidebarOpen(s => !s);
  };

  const handleSidebarSelect = (selection: PendingSelection) => {
    setPendingSelection(selection);
    setCurrentDatasetId('reset' in selection ? null : selection.dataset_id);
  };

  const handleNewDataset = () => {
    // Signal Dashboard to go back to upload state
    setPendingSelection({ reset: true });
    setCurrentDatasetId(null);
  };

  const handleDatasetIdChange = (id: number | null) => {
    setCurrentDatasetId(id);
  };

  return (
    <div className="app-container">
      {/* Sliding sidebar wrapper — controls width with CSS transition */}
      <div
        style={{
          width: sidebarOpen ? 'var(--sidebar-width)' : '0px',
          overflow: 'hidden',
          flexShrink: 0,
          transition: 'width 0.3s ease',
        }}
      >
        <Sidebar
          currentDatasetId={currentDatasetId}
          onSelect={handleSidebarSelect}
          onNewDataset={handleNewDataset}
        />
      </div>

      <div
        className="main-content"
        style={{ transition: 'width 0.3s ease' }}
      >
        <Header theme={theme} toggleTheme={toggleTheme} toggleSidebar={toggleSidebar} sidebarOpen={sidebarOpen} />
        <main className="content-area">
          <Dashboard
            pendingSelection={pendingSelection}
            onPendingConsumed={() => setPendingSelection(null)}
            onDatasetIdChange={handleDatasetIdChange}
          />
        </main>
      </div>
    </div>
  );
}

export default App;
