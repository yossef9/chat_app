import { useState, useContext } from 'react';
import ChatContext from '../context/CreateChatContext';

const ApiKeyModal = () => {
  const { apiKey, initializeOpenAI } = useContext(ChatContext);
  const [tempApiKey, setTempApiKey] = useState('');
  const [showModal, setShowModal] = useState(!apiKey);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (tempApiKey.trim()) {
      initializeOpenAI(tempApiKey.trim());
      setShowModal(false);
      setTempApiKey('');
    }
  };

  if (!showModal) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg p-6 w-96 max-w-md mx-4">
        <h2 className="text-xl font-bold mb-4">Configure OpenAI API Key</h2>
        <p className="text-gray-600 mb-4">
          To use the document chat feature, please enter your OpenAI API key.
        </p>
        
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              API Key:
            </label>
            <input
              type="password"
              value={tempApiKey}
              onChange={(e) => setTempApiKey(e.target.value)}
              placeholder="sk-..."
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              required
            />
          </div>
          
          <div className="flex space-x-3">
            <button
              type="submit"
              className="flex-1 bg-blue-600 text-white py-2 rounded-md hover:bg-blue-700"
            >
              Save API Key
            </button>
            <button
              type="button"
              onClick={() => setShowModal(false)}
              className="flex-1 bg-gray-300 text-gray-700 py-2 rounded-md hover:bg-gray-400"
            >
              Skip
            </button>
          </div>
        </form>
        
        <div className="mt-4 text-xs text-gray-500">
          <p>💡 API key will be stored in your browser and not sent anywhere else.</p>
        </div>
      </div>
    </div>
  );
};

export default ApiKeyModal;