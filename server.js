import { configDotenv } from 'dotenv';
import express from 'express';
import http from 'http';
import cors from 'cors';
import { Server } from 'socket.io';
import axios from 'axios';
import { GoogleGenerativeAI } from '@google/generative-ai';
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import morgan from 'morgan';

configDotenv();
const app = express();
app.use(cors());
app.use(express.json());
app.use(morgan('dev'));

const server = http.createServer(app);
const io = new Server(server, { cors: { origin: '*' } });

// ---- Config ----
const CHROMA_URL = process.env.CHROMA_URL || 'http://localhost:8000'; // point to Chroma server
const GEMINI_API_KEY = process.env.GOOGLE_API_KEY;

// ---- Gemini SDK ----
const genAI = new GoogleGenerativeAI(GEMINI_API_KEY);
const embeddingModel = new GoogleGenerativeAIEmbeddings({
  model: "models/embedding-001",
  apiKey: GEMINI_API_KEY,
});

// ---- Embedding ----
async function embedText(texts) {
  return Promise.all(texts.map(t => embeddingModel.embedQuery(t)));
}

// ---- Query Chroma ----
async function queryCollection(collection, query, k = 3) {
  const embedding = (await embedText([query]))[0];

  const res = await axios.post(`${CHROMA_URL}/api/v1/collections/${collection}/query`, {
    query_embeddings: [embedding],
    n_results: k,
  });

  return res.data;
}

// ---- Generate Answer with Gemini ----
async function generateAnswer(prompt) {
  const model = genAI.getGenerativeModel({ model: 'gemini-2.0-flash' });
  const result = await model.generateContent(prompt);
  return result.response.text();
}

// ---- Socket.io ----
io.on('connection', (socket) => {
  console.log('Client connected:', socket.id);

  socket.on('ask', async ({ question }) => {
    try {
      socket.emit('progress', { stage: 'retrieving', message: 'Searching knowledge base...' });

      // Query ChromaDB collection "news"
      const results = await queryCollection('news', question, 4);

      // Combine retrieved docs
      const context = results.documents[0].join('\n---\n');
      const prompt = `Answer the question based on the context:\n${context}\n\nQ: ${question}\nA:`;

      // Generate final answer with Gemini
      const answer = await generateAnswer(prompt);

      socket.emit('answer', { answer, sources: results.documents[0] });
    } catch (err) {
      console.error(err);
      socket.emit('error', { message: err.message });
    }
  });

  socket.on('disconnect', () => {
    console.log('Client disconnected:', socket.id);
  });
});

// ---- API Route for Embedding Test ----
app.post("/api/v1/embed", async (req, res) => {
  try {
    const { text } = req.body;
    if (!text) {
      return res.status(400).json({ error: "Text is required" });
    }

    const vector = await embeddingModel.embedQuery(text);
    res.json({ text, vector });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Failed to generate embeddings" });
  }
});
app.post("/api/v1/ask", async (req, res) => {
  try {
    const { question } = req.body;
    if (!question) {
      return res.status(400).json({ error: "Question is required" });
    }

    // 1. Query Chroma
    const results = await queryCollection("news", question, 4);

    // 2. Build context
    const context = results.documents[0].join("\n---\n");
    const prompt = `Answer using the context:\n${context}\n\nQ: ${question}\nA:`;

    // 3. Generate answer
    const answer = await generateAnswer(prompt);

    // 4. Save chat in MongoDB
    const chat = await prisma.chat.create({
      data: {
        question,
        answer,
        sources: results.documents[0], // optional
      },
    });

    res.json({ question, answer, sources: results.documents[0], chatId: chat.id });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Failed to process question" });
  }
});
// ---- Start Server ----
const PORT = process.env.PORT || 5100;
server.listen(PORT, () => console.log(`🚀 Gateway running at http://localhost:${PORT}`));
