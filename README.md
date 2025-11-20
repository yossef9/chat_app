# Document Chat AI

A React application that allows users to upload documents and chat with an AI based on the document's content.

## ✨ Features

- 📄 **Upload multiple file types: PDF, DOCX, TXT (max 10MB)**
- 🤖 **Smart chat: AI responds based on document content**
- 🔍 **Accurate search: AI only uses information from uploaded documents**
- 💾 **API key storage: Automatically saves OpenAI API key in localStorage**
- 📱 **Responsive: User-friendly interface on all devices**

## 🚀 Getting Started

### 1. Install and Run the Project

```bash
# Clone the project

cd Document-Chat-AI-main


# Install dependencies

npm install


# Run the development server

npm run dev

```

## 2. Configure API Key

- Open the application at http://localhost:5173/Document-Chat-AI/
- Enter your OpenAI API key when prompted
- The API key will be saved in localStorage

## 3. Upload Documents

- Click the "Choose file to upload" button or drag and drop files into the upload area
- Supported formats: PDF, DOCX, TXT
- Maximum file size: 10MB

## 4. Chat with Documents

- After successfully uploading a document, ask questions about its content
- The AI will respond based on the uploaded document's content
- The AI will cite and reference relevant sections of the document

## 🛠 Technologies Used

- **Frontend: React 18 + Vite**
- **Styling: TailwindCSS**
- **AI: OpenAI GPT-3.5-turbo**
- **File Processing:**
- **PDF: pdfjs-dist**
- **DOCX: mammoth**
- **TXT: native browser API**

## 📁 Project Structure

```
src/
├── components/     # React components
│ ├── ChatWindow.jsx    # Chat interface
│ ├── DocumentView.jsx  # Document display
│ ├── FileUpload.jsx    # File upload component
│ ├── Sidebar.jsx   # Sidebar component
│ └── ApiKeyModal.jsx   # API key input modal
├── context/    # React Context
│ ├── ChatProvider.jsx  # Main provider
│ └── CreateChatContext.jsx     # Context definition
├── services/   # Services
│ ├── openaiService.js  # OpenAI API service
│ └── fileProcessorService.js   # File processing service
├── pages/  # Pages
│ └── ChatPage.jsx  # Main page
└── layouts/    # Layouts
└── MainLayout.jsx  # Main layout
```

## 🔧 Configuration

### Environment Variables

- No .env file is required. The API key is entered directly in the application.

### Node.js Version

- Required: Node.js v18.15.0+
- Recommended: Node.js v20+ for the latest dependencies

## 🎯 Usage Guide

### Asking Effective Questions

1. **Specific questions: "What does the document say about topic X?"**
2. **Search for information: "Is there any information about Y in the document?"**
3. **Summarization: "Summarize the main content of the document"**
4. **Citations: "Quote the section discussing Z"**

### AI Response Rules

- ✅ Uses only information from the uploaded document
- ✅ Cites sources and locations within the document
- ✅ Responds with "This information is not available in the document" if no relevant content is found
- ✅ Provides concise, accurate, and clear answers

## 🔒 Security

- API key is stored locally in the browser
- No API key is sent to external servers
- Documents are processed entirely in the browser
- No document storage on servers

## 🐛 Troubleshooting

### Common Issues

1. **"Invalid API key"**

   - Verify your OpenAI API key
   - Ensure the API key has access to GPT-3.5-turbo

2. **"File too large"**

   - Reduce file size to under 10MB
   - Use a PDF compression tool if necessary

3. **"Unsupported file type"**

   - Only PDF, DOCX, and TXT are supported
   - Convert files to a supported format

4. **Node.js version errors**
   - Update Node.js to version 18.15.0+
   - Use NVM to manage multiple Node versions

## 📄 License

- MIT License - see the LICENSE file for details.

## 🤝 Contributing

- Contributions are welcome! Please create an issue or pull request.

---

- **Developed by**: Lehai
- **Version**: 1.0.0
- **Updated**: September 2025
# AI_chat_platform
