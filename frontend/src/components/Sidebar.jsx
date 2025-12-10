import { useEffect, useState, useContext } from 'react';
import FileUpload from './FileUpload';
import SessionNameModal from './SessionNameModal';
import documentService from '../services/documentService';
import ChatContext from '../context/CreateChatContext';
import sessionService from '../services/sessionService';

const Sidebar = () => {
  const [docs, setDocs] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const { selectedServerDocId, setSelectedServerDocId, activeSessionId, startNewSession, selectSession, deleteSession, sessionLoading } = useContext(ChatContext);
  const [sessions, setSessions] = useState([]);
  const [sessionsLoading, setSessionsLoading] = useState(false);
  const [sessionsError, setSessionsError] = useState('');
  const [showSessionModal, setShowSessionModal] = useState(false);

  const load = async () => {
    try {
      setLoading(true);
      setError('');
      const list = await documentService.listMine();
      setDocs(list);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  const onUploaded = () => load();
  const onDelete = async (id) => {
    try {
      await documentService.remove(id);
      await load();
    } catch (e) {
      setError(e.message);
    }
  };

  const loadSessions = async () => {
    try {
      setSessionsLoading(true);
      setSessionsError('');
      const list = await sessionService.list();
      setSessions(list);
    } catch (e) {
      setSessionsError(e.message);
    } finally {
      setSessionsLoading(false);
    }
  };

  useEffect(() => {
    loadSessions();
  }, []);

  const handleCreateSession = async (sessionName) => {
    try {
      const created = await startNewSession(sessionName);
      await loadSessions();
      await selectSession(created.id);
    } catch (e) {
      alert(e.message);
    }
  };

  const openSessionModal = () => {
    setShowSessionModal(true);
  };

  const closeSessionModal = () => {
    setShowSessionModal(false);
  };

  const handleSelectSession = async (id) => {
    try {
      await selectSession(id);
    } catch (e) {
      alert(e.message);
    }
  };

  const handleDeleteSession = async (id) => {
    if (!confirm('Delete this session?')) return;
    try {
      await deleteSession(id);
      await loadSessions();
    } catch (e) {
      alert(e.message);
    }
  };


  return (
    <div className="w-full bg-gray-50 border-r border-gray-200 flex flex-col h-full">
      <div className="p-4 border-b border-gray-200">
        <h1 className="text-xl font-bold text-gray-800">Document Chat AI</h1>
        <p className="text-sm text-gray-600 mt-1">Upload and chat with documents</p>
      </div>

      <div className="flex-1 overflow-y-auto">
        {/* File Upload Section */}
        <div className="p-4">
          <FileUpload onUploaded={onUploaded} />
        </div>

        {/* My Documents */}
        <div className="px-4 pb-4">
          <div className="flex items-center justify-between mb-2">
            <h2 className="text-sm font-semibold text-gray-800">My Documents</h2>
            <button onClick={load} className="text-xs text-blue-600 hover:text-blue-700">Refresh</button>
          </div>
          {error && (
            <div className="text-xs text-red-700 bg-red-50 border border-red-200 rounded p-2 mb-2">{error}</div>
          )}
          {loading ? (
            <div className="text-xs text-gray-500">Loading…</div>
          ) : (
            <ul className="space-y-2">
              {docs.map(doc => (
                <li key={doc.id} className="border rounded p-2 bg-white">
                  <div className="flex items-start justify-between gap-2">
                    <div className="flex-1 min-w-0">
                      <div className="text-sm font-medium text-gray-900 truncate" title={doc.filename}>{(doc.filename && doc.filename.length > 10) ? `${doc.filename.slice(0, 10)}...` : doc.filename}</div>
                      <div className="text-xs text-gray-500 mb-1">{doc.type?.toUpperCase()} · <span className={`px-1.5 py-0.5 rounded ${doc.status==='ready'?'bg-green-100 text-green-700':doc.status==='indexing'?'bg-blue-100 text-blue-700':'bg-yellow-100 text-yellow-700'}`}>
                        {doc.status === 'indexing' ? '⏳ Processing...' : (doc.status || 'uploaded')}
                      </span></div>
                      {doc.preview && (
                        <div className="text-xs text-gray-600 bg-gray-50 p-2 rounded border max-h-20 overflow-hidden" title={doc.preview}>
                          {doc.preview.length > 100 ? `${doc.preview.slice(0, 100)}...` : doc.preview}
            </div>
                      )}
                    </div>
                    <div className="flex items-center gap-1 flex-shrink-0">
                      <button onClick={() => onDelete(doc.id)} className="text-xs text-red-600 hover:text-red-700 px-2 py-1 rounded hover:bg-red-50">Delete</button>
                    </div>
                  </div>
                </li>
              ))}
              {docs.length === 0 && (
                <li className="text-xs text-gray-500">No documents yet.</li>
              )}
            </ul>
          )}
        </div>

        {/* Chat Sessions */}
        <div className="px-4 pb-6 border-t border-gray-200 pt-4">
          <div className="flex items-center justify-between mb-2">
            <h2 className="text-sm font-semibold text-gray-800">Chat Sessions</h2>
            <div className="flex items-center gap-2">
              <button onClick={loadSessions} className="text-xs text-blue-600 hover:text-blue-700">Refresh</button>
              <button 
                onClick={openSessionModal} 
                disabled={sessionLoading}
                className={`text-xs rounded px-2 py-1 transition-all duration-200 ${
                  sessionLoading 
                    ? 'text-gray-400 bg-gray-300 cursor-not-allowed' 
                    : 'text-white bg-blue-600 hover:bg-blue-700'
                }`}
              >
                {sessionLoading ? 'Loading...' : 'New'}
              </button>
            </div>
          </div>
          {sessionsError && (
            <div className="text-xs text-red-700 bg-red-50 border border-red-200 rounded p-2 mb-2">{sessionsError}</div>
          )}
          {sessionsLoading ? (
            <div className="text-xs text-gray-500">Loading…</div>
          ) : (
            <ul className="space-y-2">
              {sessions.map(s => (
                <li key={s.id} className={`border rounded p-2 bg-white ${activeSessionId===s.id ? 'ring-2 ring-indigo-500' : ''}`}>
                  <div className="flex items-center justify-between">
                    <div className="min-w-0">
                      <div className="text-sm font-medium text-gray-900 truncate flex items-center gap-2">
                        {s.name || 'Untitled Session'}
                        {activeSessionId === s.id && (
                          <span className="text-xs bg-indigo-100 text-indigo-700 px-1.5 py-0.5 rounded">Active</span>
                        )}
                      </div>
                      <div className="text-xs text-gray-500 truncate">
                        {new Date(s.updated_at).toLocaleString()}
                      </div>
                    </div>
                    <div className="flex items-center gap-1">
                      <button 
                        onClick={() => handleSelectSession(s.id)} 
                        disabled={sessionLoading}
                        className={`text-xs px-2 py-1 rounded transition-all duration-200 ${
                          sessionLoading 
                            ? 'text-gray-400 cursor-not-allowed' 
                            : activeSessionId === s.id 
                              ? 'text-indigo-600 bg-indigo-50' 
                              : 'text-indigo-600 hover:text-indigo-700 hover:bg-indigo-50'
                        }`}
                      >
                        {sessionLoading ? 'Loading...' : (activeSessionId === s.id ? 'Current' : 'Open')}
                      </button>
                      <button 
                        onClick={() => handleDeleteSession(s.id)} 
                        disabled={sessionLoading}
                        className={`text-xs px-2 py-1 rounded transition-all duration-200 ${
                          sessionLoading 
                            ? 'text-gray-400 cursor-not-allowed' 
                            : 'text-red-600 hover:text-red-700 hover:bg-red-50'
                        }`}
                      >
                        Delete
                      </button>
                    </div>
                  </div>
                </li>
              ))}
              {sessions.length === 0 && (
                <li className="text-xs text-gray-500">No sessions yet.</li>
              )}
            </ul>
          )}
        </div>
      </div>

      {/* Session Name Modal */}
      <SessionNameModal
        isOpen={showSessionModal}
        onClose={closeSessionModal}
        onConfirm={handleCreateSession}
        title="Create New Chat Session"
      />
    </div>
  );
};

export default Sidebar;