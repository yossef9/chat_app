import { useContext } from "react";
import ChatContext from "../context/CreateChatContext";
import { useState, useEffect } from "react";
import documentService from "../services/documentService";

function DocumentView() {
  const { currentDocument, clearDocument, selectedServerDocId } = useContext(ChatContext);
  const [serverPreview, setServerPreview] = useState("");
  const [loading, setLoading] = useState(false);

  const formatFileSize = (bytes) => {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
  };

  useEffect(() => {
    const load = async () => {
      if (!selectedServerDocId) {
        setServerPreview("");
        return;
      }
      try {
        setLoading(true);
        const docs = await documentService.listMine();
        const found = docs.find(d => d.id === selectedServerDocId);
        setServerPreview(found?.preview || "");
      } catch (_) {
        setServerPreview("");
      } finally {
        setLoading(false);
      }
    };
    load();
  }, [selectedServerDocId]);

  return (
    <div className="h-full flex flex-col bg-white">
      <div className="p-4 border-b border-gray-200 flex items-center justify-between">
        <h2 className="text-lg font-semibold text-gray-800">Document</h2>
        {currentDocument && (
          <button onClick={clearDocument} className="text-sm text-red-600 hover:text-red-700">
            Clear
          </button>
        )}
      </div>

      <div className="flex-1 overflow-hidden">
        {currentDocument ? (
          <div className="h-full flex flex-col">
            <div className="p-4 bg-blue-50 border-b border-blue-200">
              <h3 className="font-medium text-gray-900">{currentDocument.metadata.fileName}</h3>
              <p className="text-sm text-gray-500">
                {currentDocument.metadata.type.toUpperCase()} • {formatFileSize(currentDocument.metadata.size)}
                {currentDocument.metadata.pages && ` • ${currentDocument.metadata.pages} pages`}
              </p>
            </div>

            <div className="flex-1 p-4 overflow-y-auto">
              <div className="prose prose-sm max-w-none">
                <h4 className="text-sm font-medium text-gray-700 mb-3">Document content:</h4>
                <div className="bg-gray-50 rounded-lg p-4 text-sm text-gray-700 leading-relaxed whitespace-pre-wrap border">
                  {selectedServerDocId ? (
                    loading ? 'Loading preview…' : (serverPreview || 'No server preview available yet.')
                  ) : (
                    <>
                      {currentDocument.text.substring(0, 1000)}
                      {currentDocument.text.length > 1000 && (
                        <div className="mt-3 pt-3 border-t border-gray-200">
                          <span className="text-xs text-gray-500 italic">
                            ... and {(currentDocument.text.length - 1000).toLocaleString()} more characters
                          </span>
                        </div>
                      )}
                    </>
                  )}
                </div>
              </div>
            </div>
          </div>
        ) : (
          <div className="h-full flex items-center justify-center">
            <div className="text-center text-gray-500">
              <svg className="mx-auto h-12 w-12 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                />
              </svg>
              <p className="mt-2 text-sm">Not Any Uploaded Document</p>
              <p className="text-xs text-gray-400 mt-1">Upload file to start chat with us ^^</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default DocumentView;
