from dotenv import load_dotenv
load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI
import os


llm = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash",
    api_key= os.getenv('GEMINI_API')
)

def do_rag(query,retriever,llm,top_k=5,summarization=False,min_score=0.2):
    results = retriever.retrieve(query,top_k=top_k,score_threshold=min_score)
    if not results:
        return {'answer':'No relevant context found','sources':[],'confidence':0.0}
    

    context = "\n\n".join([doc['content'] for doc in results])

    sources = [{
        'source': doc['metadata'].get('source','unknown'),
        'page' : doc['metadata'].get('page','unknown'),
        'score': doc['similarity_score'],
    } for doc in results]

    confidence = max([doc['similarity_score'] for doc in results])

    prompt = f""" 
        Use the following context to answer the question concisely
        context:{context}
        question:{query}
    """

    response = llm.invoke([prompt.format(context=context,query=query)])
    
    output = {
        'answer' : response.content,
        'sources' : sources,
        'confidence' : confidence
    }

    if summarization:
        summary_prompt = f"Summarize the following answer in 2 sentences: \n {response.content}"
        summary_response = llm.invoke([summary_prompt])
        summary = summary_response.content
        output['summary']=summary

    return output