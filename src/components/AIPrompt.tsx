import { useState, useRef, useEffect } from 'react';
import { Sparkles, Send, History, Trash2 } from 'lucide-react';
import { useLang } from '../contexts/useLang';
import './AIPrompt.css';

const HISTORY_KEY = 'ai_prompt_history';
const MAX_HISTORY = 8;

interface AIPromptProps {
  onGenerate: (prompt: string) => void;
  columns?: string[];
}

export default function AIPrompt({ onGenerate, columns = [] }: AIPromptProps) {
  const [prompt, setPrompt] = useState('');
  const [showHistory, setShowHistory] = useState(false);
  const [history, setHistory] = useState<string[]>(
    () => JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]')
  );
  const historyRef = useRef<HTMLDivElement>(null);
  const { t } = useLang();
  const p = t.prompt;

  useEffect(() => {
    const handle = (e: MouseEvent) => {
      if (historyRef.current && !historyRef.current.contains(e.target as Node)) {
        setShowHistory(false);
      }
    };
    document.addEventListener('mousedown', handle);
    return () => document.removeEventListener('mousedown', handle);
  }, []);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!prompt.trim()) return;

    const newHistory = [prompt, ...history.filter(h => h !== prompt)].slice(0, MAX_HISTORY);
    setHistory(newHistory);
    localStorage.setItem(HISTORY_KEY, JSON.stringify(newHistory));

    onGenerate(prompt);
    setPrompt('');
  };

  const selectFromHistory = (item: string) => {
    setPrompt(item);
    setShowHistory(false);
  };

  const clearHistory = () => {
    setHistory([]);
    localStorage.removeItem(HISTORY_KEY);
    setShowHistory(false);
  };

  const getSuggestions = () => {
    if (columns.length === 0) {
      return [
        { label: 'Bar Chart', prompt: 'show top values as bar chart' },
        { label: 'Trends', prompt: 'show trend over time' },
        { label: 'Summary', prompt: 'summary of key metrics' },
      ];
    }

    const numericCols = columns.filter(c => {
      const lower = c.toLowerCase();
      return lower.includes('sales') || lower.includes('price') || lower.includes('amount') || lower.includes('count') || lower.includes('value');
    });

    const categoryCols = columns.filter(c => {
      const lower = c.toLowerCase();
      return lower.includes('category') || lower.includes('region') || lower.includes('type') || lower.includes('status') || lower.includes('name');
    });

    const firstNum = numericCols[0] || columns[0];
    const firstCat = categoryCols[0] || columns[1] || columns[0];

    const suggestions = [
      { label: p.suggestionDistribution(firstCat), prompt: p.suggestionDistributionPrompt(firstCat) },
      { label: p.suggestionAnalysis(firstNum), prompt: p.suggestionAnalysisPrompt(firstNum) },
    ];

    if (numericCols.length >= 2) {
      suggestions.push({
        label: p.suggestionCorrelation,
        prompt: p.suggestionCorrelationPrompt(numericCols[0], numericCols[1]),
      });
    } else {
      suggestions.push({
        label: p.suggestionForecast,
        prompt: p.suggestionForecastPrompt(firstNum),
      });
    }

    return suggestions;
  };

  return (
    <div className="prompt-container">
      <form onSubmit={handleSubmit} className="prompt-form">
        <div className="sparkle-wrapper">
          <Sparkles className="prompt-icon" size={20} />
        </div>
        <input
          type="text"
          className="prompt-input"
          placeholder={p.placeholder}
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
        />

        <div className="prompt-history-wrapper" ref={historyRef}>
          <button
            type="button"
            className={`prompt-history-btn${history.length > 0 ? ' has-items' : ''}`}
            onClick={() => setShowHistory(s => !s)}
            title={p.history}
          >
            <History size={16} />
            {history.length > 0 && <span className="history-badge">{history.length}</span>}
          </button>

          {showHistory && (
            <div className="history-dropdown">
              <div className="history-header">
                <span className="history-title">{p.history}</span>
                {history.length > 0 && (
                  <button type="button" className="history-clear-btn" onClick={clearHistory} title={p.clearHistory}>
                    <Trash2 size={13} />
                    {p.clearHistory}
                  </button>
                )}
              </div>
              {history.length === 0 ? (
                <div className="history-empty">{p.noHistory}</div>
              ) : (
                <ul className="history-list">
                  {history.map((item, idx) => (
                    <li key={idx}>
                      <button
                        type="button"
                        className="history-item"
                        onClick={() => selectFromHistory(item)}
                      >
                        <History size={12} className="history-item-icon" />
                        <span>{item}</span>
                      </button>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          )}
        </div>

        <button type="submit" className="prompt-submit" disabled={!prompt.trim()}>
          <Send size={18} />
        </button>
      </form>
      <div className="prompt-suggestions">
        <span className="suggestion-label">{p.suggestions}</span>
        {getSuggestions().map((s, idx) => (
          <button
            key={idx}
            type="button"
            onClick={() => setPrompt(s.prompt)}
            className="suggestion-chip"
          >
            {s.label}
          </button>
        ))}
      </div>
    </div>
  );
}
