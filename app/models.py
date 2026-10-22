from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.request
from typing import Sequence

import numpy as np

from .config import EMBEDDING_MODEL, OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT

class Embedder:
    def __init__(self):
        self.model = None
        self.name = "hashing-fallback"
        self.is_e5 = "e5" in EMBEDDING_MODEL.lower()
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(EMBEDDING_MODEL)
            self.name = EMBEDDING_MODEL
        except Exception:
            pass

    @property
    def dimension(self) -> int:
        return int(self.model.get_sentence_embedding_dimension()) if self.model else 384

    def encode(self, texts: Sequence[str], is_query: bool = False) -> np.ndarray:
        if self.model:
            prepared = list(texts)
            if self.is_e5:
                prefix = "query: " if is_query else "passage: "
                prepared = [prefix + text for text in prepared]
            return np.asarray(self.model.encode(prepared, normalize_embeddings=True, show_progress_bar=False), dtype="float32")
        vectors = np.zeros((len(texts), self.dimension), dtype="float32")
        for row, text in enumerate(texts):
            for token in re.findall(r"\w+", text.lower()):
                digest = hashlib.sha256(token.encode()).digest()
                index = int.from_bytes(digest[:4], "big") % self.dimension
                vectors[row, index] += 1.0 if digest[4] % 2 else -1.0
            norm = np.linalg.norm(vectors[row])
            if norm:
                vectors[row] /= norm
        return vectors

class Generator:
    def __init__(self):
        self.ollama_base_url = OLLAMA_BASE_URL.rstrip("/")
        self.model_name = OLLAMA_MODEL

    def answer(self, query: str, context: str, matrix_language: str, cmi: float, history: Sequence[dict] | None = None, citations: Sequence[dict] | None = None) -> str:
        history_text = "\n".join(f"{message.get('role', 'user').title()}: {message.get('content', '')}" for message in (history or [])[-12:])
        citation_instruction = ("Cite supporting passages inline using the exact format [filename · chunk N]. End with a Sources line containing every source used. Do not invent filenames or chunks."
                                if context and citations else
                                "This is a conversation-memory or casual turn. Do not cite, mention, or invent uploaded documents or sources.")
        prompt = ("Answer only from the supplied context when discussing indexed documents. Use the conversation history for personal facts and follow-up references. "
                  "Mirror the user's language mix and be concise. "
                   "Automatically detect and mirror the user's language, script, tone, and natural code-switching style (including Hinglish, Tanglish, and other multilingual combinations) from their actual message, without requiring them to explicitly request a language change; preserve Romanized Hindi or Tamil when they use it, and adapt immediately when their language mix changes. "
                  f"Matrix language: {matrix_language}; CMI: {cmi:.2f}. {citation_instruction}\n\n"
                  f"Conversation history:\n{history_text or '(no previous messages)'}\n\nContext:\n{context or '(no matching document context)'}\n\nQuestion: {query}")
        try:
            payload = json.dumps({
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": "You are a grounded multilingual code-switching conversational RAG assistant. Never claim a document fact without citing its supplied source."},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "options": {"temperature": 0.1},
            }).encode("utf-8")
            request = urllib.request.Request(
                f"{self.ollama_base_url}/api/chat",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=OLLAMA_TIMEOUT) as response:
                body = json.loads(response.read().decode("utf-8"))
            content = body.get("message", {}).get("content")
            if content:
                return content.strip()
        except Exception:
            # The app remains usable when Ollama is stopped or the model is not pulled.
            pass
        lowered = query.lower()
        for message in reversed(history or []):
            content = str(message.get("content", ""))
            match = re.search(r"\bmy name is ([A-Za-z][A-Za-z .'-]{0,60})", content, re.IGNORECASE)
            if match and ("name" in lowered or "who am i" in lowered):
                return f"Your name is {match.group(1).strip().rstrip('.!?')} .".replace(" .", ".")
        if not context and (lowered in {"hi", "hello", "hey", "thanks", "thank you"}):
            return "Hello. How can I help you?" if lowered not in {"thanks", "thank you"} else "You're welcome."
        passages = [p.strip() for p in context.split("\n\n") if p.strip()]
        if not passages:
            return "I could not find supporting information in the indexed documents."
        lead = passages[0]
        if lead.startswith("[") and "] " in lead:
            lead = lead.split("] ", 1)[1]
        if matrix_language == "hi":
            return f"Available documents ke according: {lead}"
        if matrix_language == "ta":
            return f"Indexed documents படி: {lead}"
        if cmi >= 0.2:
            return f"According to the indexed documents: {lead}"
        return f"According to the indexed documents: {lead}"
