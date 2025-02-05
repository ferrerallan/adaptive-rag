import openai
import faiss
import numpy as np
from typing import List, Dict
import json
import os
from dotenv import load_dotenv

load_dotenv()

class AdaptiveRAG:
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
        self.dimension = 1536
        self.index = faiss.IndexFlatL2(self.dimension)
        self.documents = []
        self.query_history = []
        self.feedback_history = []
        self.k_range = (2, 5)  # Min and max documents to retrieve
        self.current_k = 3
        
    def add_documents(self, docs: List[str]):
        embeddings = []
        for doc in docs:
            response = self.client.embeddings.create(
                input=doc, model="text-embedding-ada-002"
            )
            embedding = response.data[0].embedding
            embeddings.append(embedding)
            self.documents.append(doc)
            
        self.index.add(np.array(embeddings).astype('float32'))

    def _adjust_retrieval_params(self, feedback_scores: Dict) -> None:
        avg_score = sum(feedback_scores.values()) / len(feedback_scores)
        
        if avg_score < 7:  # If performance is poor, retrieve more context
            self.current_k = min(self.current_k + 1, self.k_range[1])
        elif avg_score > 9:  # If performance is very good, try reducing context
            self.current_k = max(self.current_k - 1, self.k_range[0])

    def _classify_query_type(self, question: str) -> str:
        prompt = f"""
        Classify the question type:
        Question: {question}
        
        Options:
        1. Factual (requires specific facts/data)
        2. Conceptual (requires broader understanding)
        3. Analytical (requires combining multiple pieces of information)
        
        Return only the number (1, 2, or 3).
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Classify the question type."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        return response.choices[0].message.content.strip()

    def _adjust_k_by_query_type(self, query_type: str):
        k_adjustments = {
            "1": -1,  # Factual queries need less context
            "2": 0,   # Conceptual queries use default k
            "3": 1    # Analytical queries need more context
        }
        adjustment = k_adjustments.get(query_type, 0)
        self.current_k = max(min(self.current_k + adjustment, self.k_range[1]), self.k_range[0])

    def generate_response(self, question: str, context: List[str]) -> str:
        prompt = f"""
        Context: {' '.join(context)}
        Question: {question}
        Generate a response based only on the provided context.
        """
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an adaptive assistant that learns from feedback."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content

    def evaluate_response(self, question: str, response: str, context: List[str]) -> Dict:
        eval_prompt = f"""
        Evaluate this response:
        Question: {question}
        Context: {context}
        Response: {response}
        
        You must respond with ONLY a valid JSON object in this exact format:
        {{"scores": {{"relevance": number, "context_usage": number, "accuracy": number}}, "suggestions": "string"}}
        
        The scores should be between 1-10.
        """
        try:
            evaluation = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a critical evaluator. Only return valid JSON."},
                    {"role": "user", "content": eval_prompt}
                ],
                temperature=0.1
            )
            return json.loads(evaluation.choices[0].message.content)
        except json.JSONDecodeError:
            # Fallback default evaluation
            return {
                "scores": {"relevance": 7, "context_usage": 7, "accuracy": 7},
                "suggestions": "Error parsing evaluation response"
            }

    def query(self, question: str) -> Dict:
        # Classify query and adjust parameters
        query_type = self._classify_query_type(question)
        self._adjust_k_by_query_type(query_type)
        
        # Get question embedding
        question_embedding = self.client.embeddings.create(
            input=question, model="text-embedding-ada-002"
        ).data[0].embedding

        # Retrieve documents
        D, I = self.index.search(np.array([question_embedding]).astype('float32'), self.current_k)
        context = [self.documents[i] for i in I[0]]
        
        # Generate and evaluate response
        response = self.generate_response(question, context)
        evaluation = self.evaluate_response(question, response, context)
        
        # Update parameters based on feedback
        self._adjust_retrieval_params(evaluation['scores'])
        
        # Store history
        self.query_history.append({
            'question': question,
            'query_type': query_type,
            'k_value': self.current_k,
            'scores': evaluation['scores']
        })
        
        return {
            "response": response,
            "context": context,
            "evaluation": evaluation,
            "current_k": self.current_k,
            "query_type": query_type
        }

if __name__ == "__main__":
    api = os.getenv("OPENAI_KEY")
    rag = AdaptiveRAG(api)
    
    docs = [
        "Python is a high-level programming language known for its simplicity.",
        "Python was created by Guido van Rossum in 1991.",
        "Python is widely used in data science and machine learning.",
        "Machine learning requires significant computational resources.",
        "Data science involves analyzing complex data sets.",
        "Python has extensive libraries for AI development.",
        "Python's syntax emphasizes code readability.",
        "Python supports multiple programming paradigms."
    ]
    rag.add_documents(docs)
    
    # Test different types of questions
    questions = [
        "When was Python created?",  # Factual
        "What makes Python suitable for data science?",  # Conceptual
        "How does Python's design philosophy impact its use in AI and data science?"  # Analytical
    ]
    
    for question in questions:
        result = rag.query(question)
        print(f"\nQuestion: {question}")
        print(f"Query type: {result['query_type']}")
        print(f"k value: {result['current_k']}")
        print(f"Response: {result['response']}")
        print(f"Evaluation: {result['evaluation']}")