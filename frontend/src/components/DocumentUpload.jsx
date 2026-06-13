import React, { useState, useRef } from 'react';
import { UploadCloud, CheckCircle, AlertCircle, FilePlus } from 'lucide-react';

export default function DocumentUpload({ onUploadSuccess }) {
  const [dragActive, setDragActive] = useState(false);
  const [status, setStatus] = useState({ type: '', message: '' });
  const [isUploading, setIsUploading] = useState(false);
  const inputRef = useRef(null);

  const handleDrag = function(e) {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = function(e) {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFiles(e.dataTransfer.files[0]);
    }
  };

  const handleChange = function(e) {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      handleFiles(e.target.files[0]);
    }
  };

  const handleFiles = async (file) => {
    setStatus({ type: '', message: '' });
    
    if (!file.name.endsWith('.pdf') && !file.name.endsWith('.txt')) {
      setStatus({ type: 'error', message: 'PDF or TXT only.' });
      return;
    }

    setIsUploading(true);
    setStatus({ type: 'loading', message: `Uploading...` });

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('/upload', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || 'Upload failed');
      }

      setStatus({ type: 'success', message: `Uploaded ${file.name}` });
      if (onUploadSuccess) onUploadSuccess(data.doc_id);
    } catch (error) {
      console.error(error);
      setStatus({ type: 'error', message: error.message });
    } finally {
      setIsUploading(false);
    }
  };

  const onButtonClick = () => {
    inputRef.current.click();
  };

  return (
    <div 
      className={`upload-compact ${dragActive ? "drag-active" : ""}`}
      onDragEnter={handleDrag}
      onDragLeave={handleDrag}
      onDragOver={handleDrag}
      onDrop={handleDrop}
      onClick={onButtonClick}
      title="Upload Document"
    >
      <input 
        ref={inputRef} 
        type="file" 
        style={{ display: "none" }} 
        onChange={handleChange} 
        accept=".pdf,.txt" 
      />
      
      {isUploading ? (
        <div style={{ animation: 'spin 1s linear infinite', color: 'var(--accent-color)' }}>
          <UploadCloud size={24} style={{ margin: '0 auto' }} />
        </div>
      ) : (
        <FilePlus size={24} style={{ margin: '0 auto', color: 'var(--text-secondary)' }} />
      )}
      
      <p>Drop file or click to upload</p>
      
      {status.message && (
        <div className={`upload-status status-${status.type}`}>
          {status.type === 'success' && <CheckCircle size={12} style={{ verticalAlign: 'middle', marginRight: '4px' }} />}
          {status.type === 'error' && <AlertCircle size={12} style={{ verticalAlign: 'middle', marginRight: '4px' }} />}
          {status.message}
        </div>
      )}
    </div>
  );
}
