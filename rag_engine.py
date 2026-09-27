"""
RAG Engine — Pinecone + embeddings-based retrieval for the magicpin bot.
Embeds all dataset documents into Pinecone and retrieves relevant context
for each conversation query.
"""
import os
import json
import hashlib
import logging
from typing import List, Dict, Optional, Any

from pinecone import Pinecone, ServerlessSpec
from langchain_groq import ChatGroq
from config import GROQ_API_KEY, GROQ_MODEL

logger = logging.getLogger(__name__)


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX", "magicpin-rag")
EMBEDDING_DIM = 1024 

class RAGEngine:
    """Pinecone-backed Retrieval Augmented Generation engine."""

    def __init__(self):
        self.pc: Optional[Pinecone] = None
        self.index = None
        self._ready = False

    def initialize(self, documents: List[Dict[str, Any]]):
        """
        Initialize Pinecone index and upsert document embeddings.
        Called once at startup.
        """
        if not PINECONE_API_KEY:
            logger.error("PINECONE_API_KEY not set — RAG disabled")
            return

        try:
            self.pc = Pinecone(api_key=PINECONE_API_KEY)

            
            existing = [idx.name for idx in self.pc.list_indexes()]
            if PINECONE_INDEX_NAME not in existing:
                logger.info(f"Creating Pinecone index '{PINECONE_INDEX_NAME}'...")
                self.pc.create_index(
                    name=PINECONE_INDEX_NAME,
                    dimension=EMBEDDING_DIM,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                )
                logger.info("Pinecone index created")

            self.index = self.pc.Index(PINECONE_INDEX_NAME)

            # Check if index needs population (Pinecone stats can be slow to update)
            indexed_flag = os.path.join(os.path.dirname(__file__), ".rag_indexed")
            
            stats = self.index.describe_index_stats()
            current_count = stats.get("total_vector_count", 0)

            # Only index if we haven't already locally tracked it AND pinecone says it's empty
            if not os.path.exists(indexed_flag) and current_count == 0:
                logger.info(f"Indexing {len(documents)} documents into Pinecone...")
                self._upsert_documents(documents)
                # Create the flag file to indicate successful indexing
                with open(indexed_flag, "w") as f:
                    f.write("indexed")
                logger.info("Document indexing complete")
            else:
                logger.info(f"Pinecone index already populated (found flag or {current_count} vectors), skipping upsert")

            self._ready = True
            logger.info("RAG engine initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize RAG engine: {e}")
            self._ready = False

    def _upsert_documents(self, documents: List[Dict[str, Any]]):
        """Embed and upsert documents into Pinecone in batches."""
        batch_size = 50
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            vectors = []
            for doc in batch:
                doc_id = doc["id"]
                text = doc["text"]
                embedding = self._get_embedding(text)
                if embedding:
                    vectors.append({
                        "id": doc_id,
                        "values": embedding,
                        "metadata": {
                            "type": doc.get("type", ""),
                            "text": text[:1000],  
                            **doc.get("metadata", {}),
                        }
                    })

            if vectors:
                self.index.upsert(vectors=vectors)
                logger.info(f"Upserted batch {i // batch_size + 1}: {len(vectors)} vectors")

    def _get_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding using Pinecone's inference API.
        Falls back to a deterministic hash-based embedding if unavailable.
        """
        try:
            result = self.pc.inference.embed(
                model="multilingual-e5-large",
                inputs=[text],
                parameters={"input_type": "passage", "truncate": "END"},
            )
            return result[0].values
        except Exception as e:
            logger.warning(f"Pinecone embedding failed, using hash fallback: {e}")
            return self._hash_embedding(text)

    def _hash_embedding(self, text: str) -> List[float]:
        """Deterministic hash-based embedding fallback (for when embedding API is down)."""
        import struct
        h = hashlib.sha512(text.encode("utf-8")).digest()
        
        values = []
        while len(values) < EMBEDDING_DIM:
            h = hashlib.sha512(h).digest()
            floats = struct.unpack(f"{len(h) // 4}f", h[:len(h) // 4 * 4])
            values.extend(floats)
        
        values = values[:EMBEDDING_DIM]
        norm = sum(v * v for v in values) ** 0.5
        if norm > 0:
            values = [v / norm for v in values]
        return values

    def retrieve(self, query: str, top_k: int = 5, filter_type: str = None,
                 merchant_id: str = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents from Pinecone for a given query.
        Returns list of {id, score, text, type, metadata}.
        """
        if not self._ready:
            return []

        try:
            query_embedding = self._get_query_embedding(query)
            if not query_embedding:
                return []

            
            filter_dict = {}
            if filter_type:
                filter_dict["type"] = {"$eq": filter_type}
            if merchant_id:
                filter_dict["merchant_id"] = {"$eq": merchant_id}

            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict if filter_dict else None,
            )

            retrieved = []
            for match in results.get("matches", []):
                retrieved.append({
                    "id": match["id"],
                    "score": match["score"],
                    "text": match.get("metadata", {}).get("text", ""),
                    "type": match.get("metadata", {}).get("type", ""),
                    "metadata": match.get("metadata", {}),
                })

            return retrieved

        except Exception as e:
            logger.error(f"RAG retrieval failed: {e}")
            return []

    def _get_query_embedding(self, text: str) -> Optional[List[float]]:
        """Generate query embedding."""
        try:
            result = self.pc.inference.embed(
                model="multilingual-e5-large",
                inputs=[text],
                parameters={"input_type": "query", "truncate": "END"},
            )
            return result[0].values
        except Exception as e:
            logger.warning(f"Query embedding failed: {e}")
            return self._hash_embedding(text)

    def get_context_for_conversation(self, merchant_id: str, message: str,
                                     category_slug: str = None) -> str:
        """
        Build RAG-augmented context string for a conversation.
        Retrieves relevant merchants, triggers, customers, and categories.
        """
        context_parts = []

        # 1. Get merchant-specific context
        merchant_docs = self.retrieve(
            query=f"merchant {merchant_id}",
            top_k=2,
            merchant_id=merchant_id,
        )
        if merchant_docs:
            context_parts.append("=== MERCHANT CONTEXT ===")
            for doc in merchant_docs:
                context_parts.append(doc["text"])

        # 2. Get relevant triggers
        trigger_docs = self.retrieve(
            query=message,
            top_k=3,
            filter_type="trigger",
        )
        if trigger_docs:
            context_parts.append("\n=== RELEVANT TRIGGERS ===")
            for doc in trigger_docs:
                context_parts.append(doc["text"])

        # 3. Get category context
        if category_slug:
            cat_docs = self.retrieve(
                query=f"category {category_slug}",
                top_k=1,
                filter_type="category",
            )
            if cat_docs:
                context_parts.append("\n=== CATEGORY CONTEXT ===")
                for doc in cat_docs:
                    context_parts.append(doc["text"])

        # 4. Get relevant customers for this merchant
        customer_docs = self.retrieve(
            query=f"customer for merchant {merchant_id}",
            top_k=2,
            merchant_id=merchant_id,
        )
        if customer_docs:
            context_parts.append("\n=== CUSTOMER CONTEXT ===")
            for doc in customer_docs:
                context_parts.append(doc["text"])

        return "\n".join(context_parts) if context_parts else ""

    @property
    def is_ready(self) -> bool:
        return self._ready


# Global singleton
rag_engine = RAGEngine()
