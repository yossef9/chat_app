import { useState, useContext } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import ChatContext from "../context/CreateChatContext";

const ChatWindow = () => {
  const [message, setMessage] = useState("");
  const { messages, sendMessage, loading, sessionLoading, activeSessionId, clearMessages } = useContext(ChatContext);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (message.trim()) {
      sendMessage(message);
      setMessage("");
    }
  };

  const handleClearMessages = async () => {
    if (!activeSessionId) return;
    
    if (window.confirm('Are you sure you want to clear all messages in this session? This action cannot be undone.')) {
      await clearMessages(activeSessionId);
    }
  };

  return (
    <div className="flex flex-col h-full">
      {/* Chat Header */}
      {activeSessionId && (
        <div className="flex items-center justify-between p-4 border-b border-gray-200 bg-gray-50 flex-shrink-0">
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
              <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
              </svg>
            </div>
            <div>
              <h2 className="text-lg font-semibold text-gray-900">Chat Session</h2>
              <p className="text-sm text-gray-600">{messages.length} messages</p>
            </div>
          </div>
          <button
            onClick={handleClearMessages}
            disabled={sessionLoading || loading || messages.length === 0}
            className="flex items-center space-x-2 px-3 py-2 text-sm text-red-600 hover:text-red-700 hover:bg-red-50 rounded-lg transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
            title="Clear all messages in this session"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
            </svg>
            <span>Clear Messages</span>
          </button>
        </div>
      )}

      {/* Messages Area - Scrollable */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4 min-h-0">
        {/* Session Loading Indicator */}
        {sessionLoading && (
          <div className="flex justify-center items-center h-full">
            <div className="bg-white border border-gray-200 rounded-xl p-8 shadow-sm max-w-md mx-auto">
              <div className="flex flex-col items-center space-y-4">
                <div className="flex items-center space-x-2">
                  <div className="w-3 h-3 bg-blue-400 rounded-full animate-bounce" />
                  <div className="w-3 h-3 bg-blue-400 rounded-full animate-bounce delay-100" />
                  <div className="w-3 h-3 bg-blue-400 rounded-full animate-bounce delay-200" />
                </div>
                <div className="text-center">
                  <h3 className="text-lg font-semibold text-gray-900 mb-2">Loading Session</h3>
                  <p className="text-sm text-gray-600">Fetching your chat history...</p>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Messages */}
        {!sessionLoading && messages.map((msg, idx) => (
          <div key={idx} className={`flex ${msg.sender === "user" ? "justify-end" : "justify-start"}`}>
            <div
              className={`max-w-[80%] rounded-xl p-4 ${
                msg.sender === "user" 
                  ? "bg-blue-600 text-white" 
                  : msg.sender === "system"
                  ? "bg-yellow-100 text-yellow-800 border border-yellow-200"
                  : "bg-white text-gray-800 border border-gray-200 shadow-sm"
              }`}
            >
              {msg.sender === "ai" ? (
                <div className="prose prose-sm max-w-none">
                  <ReactMarkdown 
                    remarkPlugins={[remarkGfm]}
                    components={{
                      h1: ({children}) => <h1 className="text-lg font-bold text-gray-900 mb-2">{children}</h1>,
                      h2: ({children}) => <h2 className="text-base font-semibold text-gray-800 mb-2">{children}</h2>,
                      h3: ({children}) => <h3 className="text-sm font-semibold text-gray-700 mb-1">{children}</h3>,
                      p: ({children}) => <p className="text-gray-700 mb-2 leading-relaxed">{children}</p>,
                      ul: ({children}) => <ul className="list-disc list-inside mb-2 space-y-1">{children}</ul>,
                      ol: ({children}) => <ol className="list-decimal list-inside mb-2 space-y-1">{children}</ol>,
                      li: ({children}) => <li className="text-gray-700">{children}</li>,
                      strong: ({children}) => <strong className="font-semibold text-gray-900">{children}</strong>,
                      em: ({children}) => <em className="italic text-gray-600">{children}</em>,
                      code: ({children}) => <code className="bg-gray-100 text-gray-800 px-1 py-0.5 rounded text-xs">{children}</code>,
                      blockquote: ({children}) => <blockquote className="border-l-4 border-blue-200 pl-4 italic text-gray-600 my-2">{children}</blockquote>
                    }}
                  >
                    {msg.text}
                  </ReactMarkdown>
                </div>
              ) : (
                <div className="whitespace-pre-wrap">{msg.text}</div>
              )}
            </div>
          </div>
        ))}

        {/* Welcome Message for Empty Sessions */}
        {!sessionLoading && messages.length === 0 && (
          <div className="flex justify-center items-center h-full">
            <div className="bg-gradient-to-r from-blue-50 to-purple-50 border border-blue-200 rounded-xl p-6 max-w-md mx-auto text-center">
              <div className="flex flex-col items-center space-y-3">
                <div className="w-12 h-12 bg-gradient-to-r from-blue-500 to-purple-600 rounded-full flex items-center justify-center">
                  <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                  </svg>
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-gray-900 mb-1">Welcome to your chat session!</h3>
                  <p className="text-sm text-gray-600">Start a conversation by asking questions about your documents.</p>
                </div>
              </div>
            </div>
          </div>
        )}
        {!sessionLoading && loading && (
          <div className="flex justify-start">
            <div className="bg-white border border-gray-200 rounded-xl p-4 shadow-sm">
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce" />
                <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce delay-100" />
                <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce delay-200" />
                <span className="text-sm text-gray-600 ml-2">AI is thinking...</span>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Input Form - Pinned to Bottom */}
      <form onSubmit={handleSubmit} className="p-4 border-t border-gray-200 bg-gray-50 flex-shrink-0">
        <div className="flex space-x-3">
          <input
            type="text"
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            placeholder="Ask me anything about your documents..."
            className="flex-1 rounded-xl border border-gray-300 px-4 py-3 focus:outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-200 transition-all duration-200"
            disabled={loading || sessionLoading}
          />
          <button
            type="submit"
            disabled={loading || sessionLoading || !message.trim()}
            className="bg-blue-600 text-white px-6 py-3 rounded-xl hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 font-medium"
          >
            {loading ? "Sending..." : sessionLoading ? "Loading..." : "Send"}
          </button>
        </div>
      </form>
    </div>
  );
};

export default ChatWindow;









