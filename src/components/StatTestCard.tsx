import React from 'react';

interface StatTestCardProps {
  data: {
    id: string;
    category: string;
    summary: string;
    significance_text: string;
    technical: string;
    significant: boolean;
  };
  featured?: boolean;
}

const StatTestCard: React.FC<StatTestCardProps> = ({ data, featured }) => {
  return (
    <div className={`test-result-card ${featured ? 'featured' : ''}`}>
      <div className="card-top">
        <span className="insight-category">{data.category}</span>
        <span className={`sig-badge ${data.significant ? 'sig-yes' : 'sig-no'}`}>
          {data.significant ? 'Signal detected' : 'No clear signal'}
        </span>
      </div>
      <p className="test-summary">{data.summary}</p>
      <div className="card-bottom">
        <details className="technical-details">
          <summary>Advanced details</summary>
          <div className="technical-content">
            <span className="sig-text">{data.significance_text}</span>
            <code>{data.technical}</code>
          </div>
        </details>
      </div>
    </div>
  );
};

export default StatTestCard;
