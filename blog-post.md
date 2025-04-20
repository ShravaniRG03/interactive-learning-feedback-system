# Interactive Learning and Feedback System: A Capstone Project for Gen AI Intensive Course

## Introduction

For my Gen AI Intensive Course Capstone 2025Q1, I built an Interactive Learning and Feedback System that leverages generative AI capabilities to create a personalized learning experience. This project demonstrates how AI can transform traditional learning approaches by providing dynamic questioning, immediate feedback, and personalized performance analysis.

## The Problem: Personalized Learning at Scale

Traditional education systems often struggle with:
- Providing personalized feedback to individual learners
- Adapting to different knowledge levels and learning paces
- Offering immediate assessment and correction
- Identifying subject-specific strengths and weaknesses

These challenges are especially pronounced in self-learning environments or resource-constrained educational settings. I wanted to create a solution that would make personalized learning more accessible.

## The Solution: Interactive AI-Powered Learning System

My capstone project addresses these challenges by creating an interactive system that:
1. Generates relevant questions from educational content
2. Evaluates user responses with immediate feedback
3. Provides correct answers when needed
4. Analyzes performance across knowledge domains
5. Identifies specific areas of strength and improvement

To demonstrate the project implementation, the system is built around environmental sustainability topics but can be adapted to any educational content.

## Implementation: How the systme is built using Gen AI

### Technology Stack
- **Google Gemini API**: For content generation and response evaluation
- **ChromaDB**: Vector database for semantic search and document retrieval
- **Python**: Core programming language

### Key Gen AI Capabilities Utilized

#### 1. Document Understanding
The system processes documents to generate contextually relevant questions. Here's how document management is implemented:

```python
# Define the document corpus
DOCUMENT1 = "Sustainable Agriculture Practices: Implementing crop rotation, cover cropping, and reduced tillage to enhance soil health and reduce reliance on chemical fertilizers and pesticides. These practices improve biodiversity and carbon sequestration in agricultural lands."
DOCUMENT2 = "Renewable Energy Integration: Transitioning to solar, wind, and geothermal energy sources to power homes and industries, reducing dependence on fossil fuels. Smart grid technologies optimize energy distribution and storage, minimizing waste."

documents = [DOCUMENT1, DOCUMENT2, ...] # Additional documents omitted for brevity
```

#### 2. Embeddings
The system utilizes a custom embedding function for ChromaDB that leverages Gemini's text-embedding-004 model:

```python
class GeminiEmbeddingFunction(EmbeddingFunction):
    document_mode = True

    @retry.Retry(predicate=is_retriable)
    def __call__(self, input: Documents) -> Embeddings:
        embedding_task = "retrieval_document" if self.document_mode else "retrieval_query"
        response = client.models.embed_content(
            model="models/text-embedding-004",
            contents=input,
            config=types.EmbedContentConfig(task_type=embedding_task),
        )
        return [e.values for e in response.embeddings]

# Configure ChromaDB
DB_NAME = "environment"
embed_fn = GeminiEmbeddingFunction()
chroma_client = Client()
db = chroma_client.get_or_create_collection(name=DB_NAME, embedding_function=embed_fn)
db.add(documents=documents, ids=[str(i) for i in range(len(documents))])
```

#### 3. Few-Shot Prompting
The system employs few-shot prompting to ensure consistent feedback style:

```python
few_shot_prompt = """Provide feedback to the user based on their answer to the question, considering the provided document. The feedback should only be "Correct" or "Incorrect". If incorrect, also provide the correct answer without explanation.

Question: What are three sustainable agriculture practices mentioned in the document?
User Response: crop rotation, cover cropping, reduce tillage
Feedback: Correct

Question: {question}
User Response: {user_response}
Feedback: """
```

#### 4. Structured Output/JSON Mode/Controlled Generation
The system produces structured output in JSON format for the performance summary:

```python
overall_summary = {"correct": 0, "incorrect": 0, "areas_of_strength": "", "areas_of_improvement": ""}
for result in results:
    if "Correct" in result["feedback"] and not "Incorrect" in result["feedback"]:
        overall_summary["correct"] += 1
    else:
        overall_summary["incorrect"] += 1

if strengths:
    strength_topics = ", ".join(strengths.keys())
    overall_summary["areas_of_strength"] = f"Your strengths appear to be in the areas of: {strength_topics}."

# Output as structured JSON
print("\n--- Overall Performance Summary (JSON) ---")
print(json.dumps(overall_summary, indent=4))
```

#### 5. Retrieval Augmented Generation (RAG)
The system implements RAG to retrieve relevant document contexts for question-answering:

```python
embed_fn.document_mode = False
query = "What is reforestation and aforestation?"
result = db.query(query_texts=[query], n_results=1)
[all_passages] = result["documents"]

# The retrieved passages are then used as context for generating questions and evaluating responses
```

## Limitations and Future Possibilities

### Current Limitations
1. **Content Scope**: The system is currently limited to the provided documents. External knowledge is not integrated.
2. **Assessment Complexity**: The feedback is binary (correct/incorrect) and doesn't capture partially correct answers.
3. **Question Generation**: Questions are generated based on available content, which can sometimes lead to repetition.
4. **Language Variability**: The system may struggle with synonyms or alternate phrasing in user responses.

### Future Possibilities
1. **Adaptive Learning Paths**: Customizing content difficulty based on user performance.
2. **Multi-modal Learning**: Incorporating images, audio, and video content for diverse learning styles.
3. **Conversational Learning**: Evolving into a dialogue-based system that can answer follow-up questions.
4. **Knowledge Graph Integration**: Creating connections between concepts for a more comprehensive learning experience.
5. **Collaborative Learning**: Enabling peer-to-peer learning through shared sessions and competitions.

## Conclusion

This Interactive Learning and Feedback System demonstrates how generative AI can be leveraged to create personalized, responsive educational experiences. By combining document understanding, embeddings, RAG, few-shot prompting, and structured output, the system provides immediate, contextually relevant feedback while tracking overall performance.

The project showcases the potential of generative AI to democratize access to personalized education. As AI technology continues to evolve, such systems can become increasingly sophisticated, adapting to individual learning styles and needs while providing rich, interactive learning experiences.

While the current implementation is focused on environmental sustainability topics, the framework is adaptable to virtually any educational content, making it a versatile tool for self-directed learning, educational institutions, or corporate training programs.
