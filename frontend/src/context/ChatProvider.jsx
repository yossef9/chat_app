import { useState, useContext, useEffect } from "react";
import ChatContext from "./CreateChatContext";
import { useAuth } from "./AuthProvider";
import openAIService from "../services/openaiService";
import chatService from "../services/chatService";
import sessionService from "../services/sessionService";

const ChatProvider = ({ children }) => {
  const [messages, setMessages] = useState([]); // Store messages
  const [loading, setLoading] = useState(false); // Loading state
  const [sessionLoading, setSessionLoading] = useState(false); // Session loading state
  const [currentDocument, setCurrentDocument] = useState(null); // Current document
  const [selectedServerDocId, setSelectedServerDocId] = useState(null); // Selected server doc id
  const [errorBanner, setErrorBanner] = useState("");
  const [activeSessionId, setActiveSessionId] = useState(null);
  const { user } = useAuth();
  // API key is now handled by the server

  // Helper function to clean duplicate content within a message
  const cleanMessageContent = (text) => {
    // Split by double newlines to identify potential duplicate sections
    const sections = text.split('\n\n');
    const uniqueSections = [];
    const seenSections = new Set();
    
    for (const section of sections) {
      const normalizedSection = section.trim().toLowerCase();
      if (!seenSections.has(normalizedSection) && section.trim()) {
        seenSections.add(normalizedSection);
        uniqueSections.push(section.trim());
      }
    }
    
    return uniqueSections.join('\n\n');
  };

  // Helper function to deduplicate messages
  const deduplicateMessages = (messages) => {
    const seen = new Set();
    return messages.filter(msg => {
      // Clean the message content first
      const cleanedText = cleanMessageContent(msg.text);
      
      // Create a more robust key by normalizing the text content
      const normalizedText = cleanedText
        .replace(/\s+/g, ' ') // Replace multiple spaces with single space
        .replace(/\n+/g, '\n') // Replace multiple newlines with single newline
        .trim()
        .toLowerCase();
      
      const key = `${normalizedText}-${msg.sender}`;
      if (seen.has(key)) {
        return false;
      }
      seen.add(key);
      return true;
    }).map(msg => ({
      ...msg,
      text: cleanMessageContent(msg.text)
    }));
  };

  // Reset chat state when user changes
  useEffect(() => {
    if (user) {
      // User logged in - reset all chat state
      setMessages([]);
      setActiveSessionId(null);
      setSelectedServerDocId(null);
      setCurrentDocument(null);
      setErrorBanner("");
    } else {
      // User logged out - clear everything
      setMessages([]);
      setActiveSessionId(null);
      setSelectedServerDocId(null);
      setCurrentDocument(null);
      setErrorBanner("");
    }
  }, [user?.id]); // Reset when user ID changes

  // Initialize OpenAI with API key
  // OpenAI initialization is now handled by the server

  // Upload document
  const uploadDocument = (document) => {
    setCurrentDocument(document);
    // Add success upload message
    setMessages(prev => deduplicateMessages([...prev, {
      text: `✅ Document "${document.metadata.fileName}" uploaded successfully! You can start asking questions about the document.`,
      sender: "system"
    }]));
  };

  // Clear document
  const clearDocument = () => {
    setCurrentDocument(null);
    setSelectedServerDocId(null);
    setMessages(prev => [...prev, {
      text: "🗑️ Document cleared. You can upload a new document.",
      sender: "system"
    }]);
  };

  // Check if message is a simple greeting
  const isGreeting = (msg) => {
    const greetingPatterns = [
      /^(hi|hello|hey|good morning|good afternoon|good evening)$/i,
      /^how are you/i,
      /^what's up/i,
      /^how's it going/i,
      /^how do you do/i,
      /^greetings/i
    ];
    return greetingPatterns.some(pattern => pattern.test(msg.trim()));
  };

  // Send message
  const sendMessage = async (message) => {
    
    if (!message.trim()) return;

    // API key is now handled by the server

    // Add user message to local state
    setMessages(prev => deduplicateMessages([...prev, { text: message, sender: "user" }]));
    setLoading(true);

    // Handle greeting responses
    if (isGreeting(message)) {
      const greetingResponse = `# 👋 Hello! I'm Your Document Analysis Assistant

I'm here to help you analyze and understand your uploaded documents. Here's what I can do for you:

## 📋 **Document Analysis Services**
- **Summarize** documents and extract key information
- **Answer specific questions** about document content
- **Compare** information across multiple documents
- **Explain complex concepts** in simple terms
- **Find specific details** within your documents

## 🚀 **Getting Started**
Simply ask me questions about your documents, such as:
- "Summarize the main points of [document name]"
- "What are the key findings in this document?"
- "Explain the methodology used in this research"
- "What are the conclusions of this study?"

**What would you like to know about your documents?**`;
      
      if (activeSessionId) {
        // Add greeting to session
        const greetingMessage = { 
          role: "assistant", 
          text: greetingResponse,
          created_at: new Date().toISOString(),
          sources: null
        };
        await sessionService.appendMessage(activeSessionId, greetingMessage);
      }
      
      setMessages(prev => deduplicateMessages([...prev, { text: greetingResponse, sender: "ai" }]));
      setLoading(false);
      return;
    }

    try {
      let responseText;
      
      // Always use backend RAG chat (searches all user documents)
      try {
          // If an active session exists, append there to persist history
          if (activeSessionId) {
            // Add user message to session
            const userMessage = { 
              role: "user", 
              text: message,
              created_at: new Date().toISOString(),
              sources: null
            };
            await sessionService.appendMessage(activeSessionId, userMessage);
          
          // Get AI response from backend (searches all documents)
          const result = await chatService.ask(message, null); // null = search all documents
          
          // Check if the response already includes sources to avoid duplication
          let responseText = result.answer || '';
          if (result?.sources?.length && !responseText.includes('Sources:')) {
            const sourcesSuffix = `\n\nSources:\n${result.sources.map(s => `- ${s.filename} (#${s.chunk_index})`).join('\n')}`;
            responseText = `${responseText}${sourcesSuffix}`.trim();
          }
          
          // Add AI message to session
          const aiMessage = { 
            role: "assistant", 
            text: responseText, 
            created_at: new Date().toISOString(),
            sources: result?.sources || null
          };
          await sessionService.appendMessage(activeSessionId, aiMessage);
          
          // Add AI response to local state
          setMessages(prev => deduplicateMessages([...prev, { text: responseText, sender: "ai" }]));
          setLoading(false);
          return;
        }
        
        // No active session - direct backend call
        const result = await chatService.ask(message, null); // null = search all documents
        
        // Check if the response already includes sources to avoid duplication
        let responseText = result.answer || '';
        if (result?.sources?.length && !responseText.includes('Sources:')) {
          const sourcesSuffix = `\n\nSources:\n${result.sources.map(s => `- ${s.filename} (#${s.chunk_index})`).join('\n')}`;
          responseText = `${responseText}${sourcesSuffix}`.trim();
        }
        
        setMessages(prev => deduplicateMessages([...prev, { text: responseText, sender: "ai" }]));
        setLoading(false);
        return;
      } catch (e) {
        console.error('Backend RAG failed:', e);
        
        // Handle different error types
        if (e.message?.includes('401') || e.message?.includes('Unauthorized')) {
          responseText = "❌ Authentication failed. Please log in again.";
        } else if (e.message?.includes('429')) {
          responseText = "❌ API rate limit exceeded. Please try again later.";
        } else if (e.message?.includes('500')) {
          responseText = "❌ AI service temporarily unavailable. Please try again in a moment.";
        } else if (e.message?.includes('Server API key not configured')) {
          responseText = "❌ Server configuration error. Please contact support.";
        } else {
          responseText = `❌ Error: ${e.message}`;
        }
      }

      // Add AI response
      setMessages(prev => deduplicateMessages([...prev, { text: responseText, sender: "ai" }]));
    } catch (error) {
      console.error('Chat error:', error);
      
      // Determine error type and provide helpful messages
      let errorMessage = error.message || 'Unexpected error';
      if (error.message?.includes('401')) {
        errorMessage = 'Authentication failed. Please log in again.';
      } else if (error.message?.includes('429')) {
        errorMessage = 'API rate limit exceeded. Please try again later.';
      } else if (error.message?.includes('network') || error.message?.includes('fetch')) {
        errorMessage = 'Network error. Please check your connection.';
      } else if (error.message?.includes('Server API key not configured')) {
        errorMessage = 'Server configuration error. Please contact support.';
      }
      
      setMessages(prev => [...prev, {
        text: `❌ Error: ${errorMessage}`,
        sender: "system"
      }]);
      setErrorBanner(errorMessage);
      setTimeout(() => setErrorBanner(""), 8000);
    } finally {
      setLoading(false);
    }
  };

  // Sessions helpers exposed to UI
  const startNewSession = async (name) => {
    const created = await sessionService.create(name || null, []); // Empty document_ids = use all documents
    setActiveSessionId(created.id);
    setMessages([]);
    return created;
  };

  const selectSession = async (sessionId) => {
    try {
      setSessionLoading(true);
      // Clear messages immediately to prevent duplicates
      setMessages([]);
      
      const session = await sessionService.get(sessionId);
      setActiveSessionId(session.id);
      
      // Clear any error states
      setErrorBanner("");
      
      // Derive chat window messages from session
      const mapped = (session.messages || []).map(m => ({ 
        text: m.text, 
        sender: m.role === 'user' ? 'user' : (m.role === 'assistant' ? 'ai' : 'system') 
      }));
      
      // Deduplicate messages to prevent duplicates
      const deduplicatedMessages = deduplicateMessages(mapped);
      
      console.log('Loading session:', session.id, 'with', deduplicatedMessages.length, 'messages');
      setMessages(deduplicatedMessages);
      
      // No need to set selectedServerDocId since we use all documents
      return session;
    } catch (error) {
      console.error('Error loading session:', error);
      setMessages([{ text: `❌ Error loading session: ${error.message}`, sender: "system" }]);
      setErrorBanner(`Failed to load session: ${error.message}`);
      throw error;
    } finally {
      setSessionLoading(false);
    }
  };

  const deleteSession = async (sessionId) => {
    await sessionService.remove(sessionId);
    if (activeSessionId === sessionId) {
      setActiveSessionId(null);
      setMessages([]);
    }
  };

  const clearMessages = async (sessionId) => {
    if (!sessionId) return;
    
    try {
      await sessionService.clearMessages(sessionId);
      // Clear local messages if this is the active session
      if (activeSessionId === sessionId) {
        setMessages([]);
      }
    } catch (error) {
      console.error('Error clearing messages:', error);
      setErrorBanner(`Failed to clear messages: ${error.message}`);
      setTimeout(() => setErrorBanner(""), 5000);
    }
  };

  // Legacy function for compatibility
  const receiveMessage = (message) => {
    setMessages(prevMessages => deduplicateMessages([...prevMessages, { text: message, sender: "ai" }]));
  };

  // Provide values and functions to the entire application
  return (
    <ChatContext.Provider value={{ 
      messages, 
      loading, 
      sessionLoading,
      currentDocument,
      selectedServerDocId,
      activeSessionId,
      sendMessage, 
      receiveMessage, 
      uploadDocument,
      clearDocument,
      setSelectedServerDocId,
      errorBanner,
      startNewSession,
      selectSession,
      deleteSession,
      clearMessages,
    }}>
      {children}
    </ChatContext.Provider>
  );
};

export default ChatProvider;
