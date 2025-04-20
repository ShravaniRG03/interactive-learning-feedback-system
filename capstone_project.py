# Cell 1 : 
!pip install -qU "google-genai==1.7.0" "chromadb==0.6.3"

# Cell 2: Import necessary modules
from google import genai
from google.genai import types
from IPython.display import Markdown, display
from kaggle_secrets import UserSecretsClient
from chromadb import Documents, EmbeddingFunction, Embeddings, Client
from google.api_core import retry
import json

# Cell 3: Configure Google Gemini API
GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
client = genai.Client(api_key=GOOGLE_API_KEY)

# Cell 4: Define the document corpus
DOCUMENT1 = "Sustainable Agriculture Practices: Implementing crop rotation, cover cropping, and reduced tillage to enhance soil health and reduce reliance on chemical fertilizers and pesticides. These practices improve biodiversity and carbon sequestration in agricultural lands."
DOCUMENT2 = "Renewable Energy Integration: Transitioning to solar, wind, and geothermal energy sources to power homes and industries, reducing dependence on fossil fuels. Smart grid technologies optimize energy distribution and storage, minimizing waste."
DOCUMENT3 = "Urban Green Spaces: Creating and expanding urban parks, green roofs, and vertical gardens to mitigate the urban heat island effect, improve air quality, and provide habitats for wildlife. These spaces also enhance community well-being and offer recreational opportunities."
DOCUMENT4 = "Waste Reduction and Circular Economy: Promoting recycling, composting, and reducing single-use plastics to minimize landfill waste. Implementing closed-loop systems where materials are reused and repurposed, reducing the need for new resource extraction."
DOCUMENT5 = "Water Conservation Strategies: Implementing efficient irrigation techniques, rainwater harvesting, and greywater recycling to conserve water resources. Protecting and restoring wetlands and watersheds to maintain natural water filtration and storage."
DOCUMENT6 = "Reforestation and Afforestation Initiatives: Planting trees and restoring forests to absorb carbon dioxide, enhance biodiversity, and prevent soil erosion. These initiatives also contribute to watershed protection and climate regulation."
DOCUMENT7 = "Carbon Capture and Storage Technologies: Developing and deploying technologies to capture carbon dioxide emissions from industrial sources and power plants, storing it underground to prevent atmospheric release. These technologies play a crucial role in mitigating climate change."
DOCUMENT8 = "Biodiversity Preservation: Protecting and restoring natural habitats, including forests, wetlands, and coral reefs, to conserve biodiversity. Implementing sustainable fishing and logging practices to minimize ecosystem disruption."

documents = [DOCUMENT1, DOCUMENT2, DOCUMENT3, DOCUMENT4, DOCUMENT5, DOCUMENT6, DOCUMENT7, DOCUMENT8]

# Cell 5: Configure retry mechanism for API calls
is_retriable = lambda e: (isinstance(e, genai.errors.APIError) and e.code in {429, 503})


# Cell 6: Define Gemini Embedding Function for ChromaDB
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

DB_NAME = "environment"
embed_fn = GeminiEmbeddingFunction()
chroma_client = Client()
db = chroma_client.get_or_create_collection(name=DB_NAME, embedding_function=embed_fn)
db.add(documents=documents, ids=[str(i) for i in range(len(documents))])

embed_fn.document_mode = False
query = "What is reforestation and aforestation?"
result = db.query(query_texts=[query], n_results=1)
[all_passages] = result["documents"]

display(Markdown(all_passages[0]))



# Cell 7: Define few-shot prompt for feedback generation
few_shot_prompt = """Provide feedback to the user based on their answer to the question, considering the provided document. The feedback should only be "Correct" or "Incorrect". If incorrect, also provide the correct answer without explanation.

Question: What are three sustainable agriculture practices mentioned in the document?
User Response: crop rotation, cover cropping, reduce tillage
Feedback: Correct

Question: How does transitioning to renewable energy sources help the environment according to the document?
User Response: It makes energy cheaper.
Feedback: Incorrect. Correct answer: Transitioning to solar, wind, and geothermal energy reduces dependence on fossil fuels, minimizing waste and environmental impact.

Question: What is the purpose of urban green spaces as described in the document?
User Response: To have parks in cities.
Feedback: Incorrect. Correct answer: To mitigate the urban heat island effect, improve air quality, and provide habitats for wildlife, in addition to enhancing community well-being and offering recreational opportunities.

Question: {question}
User Response: {user_response}
Feedback: """

# Cell 8: Define functions for asking questions, getting user responses, and generating feedback
def ask_question(context):
    response = client.models.generate_content(
        model='gemini-2.0-flash',
        config=types.GenerateContentConfig(
            temperature=0.1,
            top_p=1,
            max_output_tokens=250,
        ),
        contents=[f"Based on the following document, ask one question:\n\n{context}"]
    )
    return response.text

def get_user_response(question):
    return input(f"{question}\n\nYour Response:\n")

def get_feedback(prompt, question, user_response, context):
    formatted_prompt = prompt.format(question=question, user_response=user_response)
    response = client.models.generate_content(
        model='gemini-2.0-flash',
        config=types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=250,
        ),
        contents=[formatted_prompt + "\n\nDocument:\n" + context + "\n\nFeedback:"]
    )
    return response.text


# Cell 9: Implement the question-answering loop and overall summary
import random

while True:
    try:
        num_questions = int(input("Enter the number of questions you'd like to answer: "))
        if num_questions > 0:
            break
        else:
            print("Please enter a positive number of questions.")
    except ValueError:
        print("Invalid input. Please enter a number.")

num_questions = min(num_questions, len(documents))
results = []
strengths = {}
improvements = {}

for i in range(num_questions):
    print(f"\n--- Question {i+1} ---")
    document_indices = list(range(len(documents)))
    random.shuffle(document_indices)
    current_document_index = document_indices[i % len(documents)] # Use modulo to cycle through if num_questions > len(documents)
    current_document = documents[current_document_index]
    current_document_title = current_document.split(":")[0].strip()
    ai_question = ask_question(current_document)
    display(Markdown(ai_question))
    user_answer = get_user_response(ai_question)
    feedback_text = get_feedback(few_shot_prompt, ai_question, user_answer, current_document)
    print("\nFeedback :")
    display(Markdown(feedback_text))
    results.append({"question": ai_question, "user_answer": user_answer, "feedback": feedback_text, "topic": current_document_title})
    if "Correct" in feedback_text and not "Incorrect" in feedback_text:
        strengths[current_document_title] = strengths.get(current_document_title, 0) + 1
    else:
        improvements[current_document_title] = improvements.get(current_document_title, 0) + 1

overall_summary = {"correct": 0, "incorrect": 0, "areas_of_strength": "", "areas_of_improvement": ""}
for result in results:
    if "Correct" in result["feedback"] and not "Incorrect" in result["feedback"]:
        overall_summary["correct"] += 1
    else:
        overall_summary["incorrect"] += 1

if strengths:
    strength_topics = ", ".join(strengths.keys())
    overall_summary["areas_of_strength"] = f"Your strengths appear to be in the areas of: {strength_topics}."
else:
    overall_summary["areas_of_strength"] = "Based on your responses so far, no specific strengths have been identified yet."

if improvements:
    improvement_topics = ", ".join(improvements.keys())
    overall_summary["areas_of_improvement"] = f"Areas where you might want to focus more include: {improvement_topics}."
else:
    overall_summary["areas_of_improvement"] = "Based on your responses so far, no specific areas for improvement have been identified yet."

print("\n--- Overall Performance Summary (JSON) ---")
print(json.dumps(overall_summary, indent=4))
