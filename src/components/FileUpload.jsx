import { useState, useContext } from 'react';
import ChatContext from '../context/CreateChatContext';
import fileProcessorService from '../services/fileProcessorService';
import documentService from '../services/documentService';

const FileUpload = ({ onUploaded }) => {
  const [isUploading, setIsUploading] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const { uploadDocument } = useContext(ChatContext);

  const handleFiles = async (files) => {
    if (files.length === 0) return;

    const file = files[0];
    
    // Check file type
    if (!fileProcessorService.isFileSupported(file)) {
      alert('File type not supported. Only PDF, DOCX, TXT are supported.');
      return;
    }

    // Check file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
      alert('File too large. Maximum size is 10MB.');
      return;
    }

    setIsUploading(true);
    
    try {
      // Upload to backend
      await documentService.upload(file);
      if (onUploaded) onUploaded();

      // Optional: keep local processing to show preview text in chat panel
      try {
        const processedFile = await fileProcessorService.processFile(file);
        uploadDocument(processedFile);
      } catch (_) {
        // ignore local processing errors
      }
    } catch (error) {
      console.error('Upload error:', error);
      alert('Error uploading file: ' + error.message);
    } finally {
      setIsUploading(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFiles([e.dataTransfer.files[0]]);
    }
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleChange = (e) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      handleFiles([e.target.files[0]]);
    }
  };

  return (
    <div className="mb-4">
      <div
        className={`relative border-2 border-dashed rounded-lg p-6 text-center transition-colors ${ 
          dragActive 
            ? 'border-blue-500 bg-blue-50' 
            : 'border-gray-300 hover:border-gray-400'
        } ${isUploading ? 'opacity-50 pointer-events-none' : ''}`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
      >
        <input
          type="file"
          id="file-upload"
          accept=".pdf,.docx,.txt"
          onChange={handleChange}
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
          disabled={isUploading}
        />
        
        <div className="space-y-2">
          {isUploading ? (
            <div className="flex items-center justify-center space-x-2">
              <div className="w-5 h-5 border-2 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
              <span className="text-blue-600">Uploading file...</span>
            </div>
          ) : (
            <>
              <div className="mx-auto w-12 h-12 text-gray-400">
                <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                </svg>
              </div>
              <div>
                <label htmlFor="file-upload" className="cursor-pointer">
                  <span className="text-blue-600 hover:text-blue-700 font-medium">
                    Choose file to upload
                  </span>
                  <span className="text-gray-500"> or drag and drop here</span>
                </label>
              </div>
              <p className="text-sm text-gray-500">
                Supports PDF, DOCX, TXT (max 10MB)
              </p>
            </>
          )}
        </div>
      </div>
    </div>
  );
};

export default FileUpload;