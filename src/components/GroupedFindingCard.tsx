import React from 'react';

interface GroupedFindingCardProps {
  data: {
    id: string;
    category: string;
    categorical_col: string;
    affected_cols: string[];
    summary: string;
    significance_text: string;
    technical: string;
  };
}

const GroupedFindingCard: React.FC<GroupedFindingCardProps> = ({ data }) => {
  return (
    <div className="test-result-card featured grouped-impact">
      <div className="card-top">
        <span className="insight-category">{data.category}</span>
        <span className="sig-badge sig-yes">✓ High Strategic Impact</span>
      </div>
      <p className="test-summary">{data.summary}</p>
      <div className="impact-tags">
        {data.affected_cols.map(col => (
          <span key={col} className="impact-tag">{col}</span>
        ))}
      </div>
      <div className="card-bottom">
        <details className="technical-details">
          <summary>Consolidated Data</summary>
          <div className="technical-content">
            <span className="sig-text">{data.significance_text}</span>
            <code>{data.technical}</code>
          </div>
        </details>
      </div>
    </div>
  );
};

export default GroupedFindingCard;
