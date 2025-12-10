import OpenAI from 'openai';
import Groq from 'groq-sdk';

class OpenAIService {
  constructor() {
    this.client = null;
    this.apiKey = null;
    this.provider = 'openai'; // 'openai' | 'groq'
  }

  // Remove whitespace and zero-width/non-ascii characters
  sanitizeKey(raw) {
    if (!raw) return '';
    return String(raw)
      .replace(/[\u200B-\u200D\uFEFF]/g, '') // zero-width chars
      .replace(/\s+/g, '') // all whitespace
      .replace(/[^\x00-\x7F]/g, ''); // non-ascii
  }

  // Initialize with API key
  initialize(apiKey) {
    this.apiKey = this.sanitizeKey(apiKey);
    // Detect provider
    if (this.apiKey.toLowerCase().startsWith('gsk_')) {
      this.provider = 'groq';
      this.client = new Groq({ apiKey: this.apiKey, dangerouslyAllowBrowser: true });
    } else {
      this.provider = 'openai';
      this.client = new OpenAI({
        apiKey: this.apiKey,
        dangerouslyAllowBrowser: true
      });
    }
  }

  // Reset client (used when API key removed)
  reset() {
    this.client = null;
    this.apiKey = null;
    this.provider = 'openai';
  }

  async testKey(apiKey) {
    const cleaned = this.sanitizeKey(apiKey);
    try {
      if (cleaned.toLowerCase().startsWith('gsk_')) {
        const groq = new Groq({ apiKey: cleaned, dangerouslyAllowBrowser: true });
        await groq.models.list();
        return { ok: true };
      } else {
        const client = new OpenAI({ apiKey: cleaned, dangerouslyAllowBrowser: true });
        await client.models.list();
        return { ok: true };
      }
    } catch (e) {
      return { ok: false, message: e?.message || 'Key test failed' };
    }
  }

  // Check if initialized
  isInitialized() {
    return this.client !== null;
  }

  // Chat with document context
  async chatWithDocument(message, documentContent, chatHistory = []) {
    if (!this.isInitialized()) {
      throw new Error('OpenAI service not initialized. Please provide API key.');
    }

    try {
      // Create system prompt for document chat
      const systemPrompt = `You are an AI assistant specialized in document reading and comprehension.

The user has uploaded a document with the following content:
"""
${documentContent}
"""

Response rules:
1. Only use content from the above document to answer
2. If possible, specify the location in the document (page number, paragraph, section...)
3. If the information is not in the document, FIRST state clearly: "This information is not available in the document." THEN, after a short separator line (---), provide a brief general explanation based on your own knowledge, labeled as "General answer (not from the document)".
4. Answer concisely, accurately, and clearly
5. Always quote relevant text passages from the document`;

      // Create messages array
      const messages = [
        { role: 'system', content: systemPrompt },
        ...chatHistory.map(msg => ({
          role: msg.sender === 'user' ? 'user' : 'assistant',
          content: msg.text
        })),
        { role: 'user', content: message }
      ];

      if (this.provider === 'groq') {
        // Use a supported Groq model; fallback if decommissioned
        const primaryModel = 'llama-3.3-70b-versatile';
        const fallbackModel = 'llama-3.1-8b-instant';
        try {
          const response = await this.client.chat.completions.create({
            model: primaryModel,
            messages,
            temperature: 0.3,
            max_tokens: 1000,
          });
          return response.choices?.[0]?.message?.content;
        } catch (err) {
          const msg = err?.message || '';
          if (msg.includes('model') && msg.includes('decommissioned')) {
            const response = await this.client.chat.completions.create({
              model: fallbackModel,
              messages,
              temperature: 0.3,
              max_tokens: 1000,
            });
            return response.choices?.[0]?.message?.content;
          }
          throw err;
        }
      } else {
        const response = await this.client.chat.completions.create({
          model: 'gpt-3.5-turbo',
          messages,
          max_tokens: 1000,
          temperature: 0.3,
        });
        return response.choices[0].message.content;
      }
    } catch (error) {
      console.error('OpenAI API Error:', error);
      if (error.status === 401) {
        throw new Error('Invalid API key. Please check again.');
      } else if (error.status === 429) {
        throw new Error('API limit exceeded. Please try again later.');
      } else {
        throw new Error('Error occurred when calling OpenAI API: ' + error.message);
      }
    }
  }

  // Simple chat (without document)
  async simpleChat(message, chatHistory = []) {
    if (!this.isInitialized()) {
      throw new Error('OpenAI service not initialized. Please provide API key.');
    }

    try {
      const messages = [
        { role: 'system', content: 'You are a helpful AI assistant.' },
        ...chatHistory.map(msg => ({
          role: msg.sender === 'user' ? 'user' : 'assistant',
          content: msg.text
        })),
        { role: 'user', content: message }
      ];

      if (this.provider === 'groq') {
        const primaryModel = 'llama-3.3-70b-versatile';
        const fallbackModel = 'llama-3.1-8b-instant';
        try {
          const response = await this.client.chat.completions.create({
            model: primaryModel,
            messages,
            max_tokens: 1000,
            temperature: 0.7,
          });
          return response.choices?.[0]?.message?.content;
        } catch (err) {
          const msg = err?.message || '';
          if (msg.includes('model') && msg.includes('decommissioned')) {
            const response = await this.client.chat.completions.create({
              model: fallbackModel,
              messages,
              max_tokens: 1000,
              temperature: 0.7,
            });
            return response.choices?.[0]?.message?.content;
          }
          throw err;
        }
      } else {
        const response = await this.client.chat.completions.create({
          model: 'gpt-3.5-turbo',
          messages,
          max_tokens: 1000,
          temperature: 0.7,
        });
        return response.choices[0].message.content;
      }
    } catch (error) {
      console.error('OpenAI API Error:', error);
      throw new Error('Error occurred when calling OpenAI API: ' + error.message);
    }
  }
}

// Singleton instance
const openAIService = new OpenAIService();
export default openAIService;