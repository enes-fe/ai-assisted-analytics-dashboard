import { useState, useRef } from 'react';
import { UploadCloud, FileType } from 'lucide-react';
import { useToast } from './useToast';
import { useLang } from '../contexts/useLang';
import type { UploadResponse } from '../types';
import './DataUploader.css';

interface DataUploaderProps {
  onDataLoaded: (response: UploadResponse) => void;
  onUploadStart: () => void;
  onUploadError: () => void;
}

export default function DataUploader({ onDataLoaded, onUploadStart, onUploadError }: DataUploaderProps) {
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { showToast } = useToast();
  const { t } = useLang();
  const u = t.uploader;

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      await processFiles(Array.from(e.dataTransfer.files));
    }
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      await processFiles(Array.from(e.target.files));
    }
  };

  const processFiles = async (files: File[]) => {
    if (files.length > 5) {
      showToast(u.maxFiles, 'warning');
      return;
    }

    const validExtensions = ['.csv', '.json', '.xlsx', '.xls'];
    for (const file of files) {
      if (!validExtensions.some(ext => file.name.toLowerCase().endsWith(ext))) {
        showToast(u.unsupportedFormat(file.name), 'error');
        return;
      }
    }

    onUploadStart();

    const formData = new FormData();
    for (const file of files) {
      formData.append('files', file);
    }

    try {
      const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        let errorMessage = 'Backend processing failed.';
        try {
          const errorData = await response.json();
          if (typeof errorData.detail === 'string') {
            errorMessage = errorData.detail;
          } else if (errorData.detail?.message) {
            errorMessage = errorData.detail.message;
          } else {
            errorMessage = JSON.stringify(errorData.detail) || errorMessage;
          }
        } catch {
          const textError = await response.text().catch(() => '');
          errorMessage = textError && textError.length < 500
            ? textError
            : `Sunucu Hatası (${response.status}): Analiz motorunda bir sorun oluştu.`;
        }
        throw new Error(errorMessage);
      }

      const responseData = await response.json() as UploadResponse;
      showToast(u.success(files.length), 'success');
      onDataLoaded(responseData);
    } catch (err) {
      console.error('Error uploading to backend:', err);
      showToast(err instanceof Error ? err.message : u.genericError, 'error');
      onUploadError();
    }
  };

  return (
    <div className="uploader-container">
      <div
        className={`drop-zone ${isDragging ? 'dragging' : ''}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
      >
        <div className="upload-icon-wrapper">
          <UploadCloud size={48} className="upload-icon" />
        </div>
        <h3 className="upload-title">{u.title}</h3>
        <p className="upload-desc">{u.desc}</p>
        <div className="supported-formats">
          <div className="format-badge">
            <FileType size={14} /> CSV
          </div>
          <div className="format-badge">
            <FileType size={14} /> JSON
          </div>
          <div className="format-badge">
            <FileType size={14} /> EXCEL
          </div>
        </div>
        <input
          type="file"
          ref={fileInputRef}
          style={{ display: 'none' }}
          accept=".csv,.json,.xlsx,.xls"
          multiple
          onChange={handleFileChange}
        />
      </div>
    </div>
  );
}
