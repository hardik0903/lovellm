import React, { useState, useRef } from 'react';
import { UploadCloud, CheckCircle, AlertCircle, FileText } from 'lucide-react';

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
    
    // Validate file type
    if (!file.name.endsWith('.pdf') && !file.name.endsWith('.txt')) {
      setStatus({ type: 'error', message: 'Only PDF and TXT files are supported.' });
      return;
    }

    setIsUploading(true);
    setStatus({ type: 'loading', message: `Uploading ${file.name}...` });

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:8000/upload', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || 'Upload failed');
      }

      setStatus({ type: 'success', message: `Processed ${data.chunks_processed} chunks from ${file.name}` });
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
      className={`upload-container ${dragActive ? "drag-active" : ""}`}
      onDragEnter={handleDrag}
      onDragLeave={handleDrag}
      onDragOver={handleDrag}
      onDrop={handleDrop}
      onClick={onButtonClick}
    >
      <input 
        ref={inputRef} 
        type="file" 
        style={{ display: "none" }} 
        onChange={handleChange} 
        accept=".pdf,.txt" 
      />
      
      {isUploading ? (
        <div className="upload-icon">
          <div style={{ animation: 'spin 1s linear infinite' }}>
            <UploadCloud size={48} />
          </div>
        </div>
      ) : (
        <div className="upload-icon">
          <FileText size={48} />
        </div>
      )}
      
      <div className="upload-text">
        <strong>Click to upload</strong> or drag and drop<br/>
        PDF or TXT files
      </div>
      
      {status.message && (
        <div className={`upload-status status-${status.type}`}>
          {status.type === 'success' && <CheckCircle size={16} style={{ verticalAlign: 'text-bottom', marginRight: '4px' }} />}
          {status.type === 'error' && <AlertCircle size={16} style={{ verticalAlign: 'text-bottom', marginRight: '4px' }} />}
          {status.message}
        </div>
      )}
    </div>
  );
}
