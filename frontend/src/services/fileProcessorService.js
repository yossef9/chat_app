import mammoth from "mammoth";
// import * as pdfjsLib from 'pdfjs-dist';

// // Configure worker for PDF.js
// pdfjsLib.GlobalWorkerOptions.workerSrc = `//cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js`;

import * as pdfjsLib from "pdfjs-dist";
import pdfWorker from "pdfjs-dist/build/pdf.worker.min?url"; // vite-friendly import

pdfjsLib.GlobalWorkerOptions.workerSrc = pdfWorker;

class FileProcessorService {
  // Process file based on file type
  async processFile(file) {
    const fileType = this.getFileType(file);

    try {
      switch (fileType) {
        case "pdf":
          return await this.processPDF(file);
        case "docx":
          return await this.processDOCX(file);
        case "txt":
          return await this.processTXT(file);
        default:
          throw new Error(`File type not supported: ${file.type}`);
      }
    } catch (error) {
      console.error("File processing error:", error);
      throw new Error(`Error processing file: ${error.message}`);
    }
  }

  // Determine file type
  getFileType(file) {
    const fileExtension = file.name.split(".").pop().toLowerCase();
    const mimeType = file.type;

    if (mimeType === "application/pdf" || fileExtension === "pdf") {
      return "pdf";
    } else if (
      mimeType === "application/vnd.openxmlformats-officedocument.wordprocessingml.document" ||
      fileExtension === "docx"
    ) {
      return "docx";
    } else if (mimeType === "text/plain" || fileExtension === "txt") {
      return "txt";
    } else {
      return "unknown";
    }
  }

  // Process PDF file
  async processPDF(file) {
    const arrayBuffer = await file.arrayBuffer();
    const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;

    let fullText = "";

    for (let pageNum = 1; pageNum <= pdf.numPages; pageNum++) {
      const page = await pdf.getPage(pageNum);
      const textContent = await page.getTextContent();

      const pageText = textContent.items.map((item) => item.str).join(" ");

      fullText += `\\n[Page ${pageNum}]\\n${pageText}\\n`;
    }

    return {
      text: fullText.trim(),
      metadata: {
        type: "pdf",
        pages: pdf.numPages,
        fileName: file.name,
        size: file.size,
      },
    };
  }

  // Process DOCX file
  async processDOCX(file) {
    const arrayBuffer = await file.arrayBuffer();
    const result = await mammoth.extractRawText({ arrayBuffer });

    return {
      text: result.value,
      metadata: {
        type: "docx",
        fileName: file.name,
        size: file.size,
      },
    };
  }

  // Process TXT file
  async processTXT(file) {
    const text = await file.text();

    return {
      text: text,
      metadata: {
        type: "txt",
        fileName: file.name,
        size: file.size,
      },
    };
  }

  // Check if file is supported
  isFileSupported(file) {
    const supportedTypes = [
      "application/pdf",
      "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
      "text/plain",
    ];

    const supportedExtensions = ["pdf", "docx", "txt"];
    const fileExtension = file.name.split(".").pop().toLowerCase();

    return supportedTypes.includes(file.type) || supportedExtensions.includes(fileExtension);
  }

  // Format file size
  formatFileSize(bytes) {
    if (bytes === 0) return "0 Bytes";

    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));

    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
  }
}

// Singleton instance
const fileProcessorService = new FileProcessorService();
export default fileProcessorService;
