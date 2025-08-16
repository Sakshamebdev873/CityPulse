import { configDotenv } from 'dotenv';
import express from 'express';
import http from 'http';
import cors from 'cors';
import { Server } from 'socket.io';
import axios from 'axios';
import { GoogleGenerativeAI} from '@google/generative-ai';
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
configDotenv()
const app = express();
app.use(cors());
app.use(express.json());

const server = http.createServer(app);
const io = new Server(server, { cors: { origin: '*' } });

// ---- Config ----
const CHROMA_URL = process.env.CHROMA_URL || 'http://localhost:8001';
const GEMINI_API_KEY = process.env.GOOGLE_API_KEY;

// ---- Gemini SDK ----
const genAI = new GoogleGenerativeAI(GEMINI_API_KEY);
const embeddingModel = new GoogleGenerativeAIEmbeddings({
  model: "models/embedding-001",
  apiKey: GEMINI_API_KEY,
});

// ---- Embedding ----
async function embedText(texts) {
  const embeddings = await Promise.all(
    texts.map(async (t) => {
      const res = await embeddingModel.embedContent(t);
      return res.embedding.values;
    })
  );
  return embeddings;
}

// ---- Add Document to Chroma ----
async function addDocument(collection, id, text) {
  const embedding = (await embedText([text]))[0];
  await axios.post(`${CHROMA_URL}/api/v1/collections/${collection}/add`, {
    ids: [id],
    embeddings: [embedding],
    documents: [text],
  });
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

      const results = await queryCollection('news', question, 4);

      const context = results.documents[0].join('\n---\n');
      const prompt = `Answer using the context:\n${context}\n\nQ: ${question}\nA:`;

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

// ---- Start Server ----
const PORT = process.env.PORT || 5100;
server.listen(PORT, () => console.log(`🚀 Gateway running at http://localhost:${PORT}`));
