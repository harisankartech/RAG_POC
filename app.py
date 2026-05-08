from pypdf import PdfReader
from groq import Groq
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

import faiss
import numpy as np

## READ THE PDF
reader=PdfReader("Hari_AI_Engineer_Roadmap.pdf")

text=""

for page in reader.pages:
    extracted=page.extract_text()
    
    if extracted:
        text+=extracted
        


##CONVERT TEXT TO TOKENS USING TOKENIZER

tokenizer=AutoTokenizer.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2"
)
tokens=tokenizer.encode(text)


##TOKEN BASED-CHUNKING

chunk_size=200
overlap=50

chunks=[]

start=0

while(start<len(tokens)):
    end=start+chunk_size
    
    chunk_tokens=tokens[start:end]
    
    chunk_text=tokenizer.decode(
        chunk_tokens,
        skip_special_tokens=True
        )
    
    chunks.append(chunk_text)
    
    start+=chunk_size-overlap
    
print(chunks)

##LOAD EMBEDDING MODEL

model=SentenceTransformer(
    "all-MiniLM-L6-v2"
)

##MODEL CONVERT CHUNKS TO EMBEDDING VECTORS
embeddings=model.encode(
    chunks,
    normalize_embeddings=True
)

# Create a FAISS vector database
# that can store vectors of size 384
# and compare similarity
dimension=embeddings.shape[1]
index=faiss.IndexFlatIP(dimension)

#ADD EMBEDDINGS TO VECTOR DATABASE CALLED 'index' HERE
index.add(
    np.array(embeddings,dtype=np.float32)
    )

while(True):
    query=input("Ask Your Question: ")

##CONVERT QUERY TO EMBEDDING VECTORS
    if query.lower() == "exit":
        print("Goodbye")
        break

    query_embeddings=model.encode(
    [query],
    normalize_embeddings=True
    )

##SEMANTIC SEARCH
#The search will give top k=3 similarity scores and the indices as output

    k=3

    scores,indices=index.search(
        np.array(query_embeddings,dtype=np.float32),
        k
    )

##RETRIEVE THE MATCHING CHUNK THROUGH LOOPING THROUGH INDICES
    retrieved_chunks=[]

    for idx in indices[0]:
        retrieved_chunks.append(
            chunks[idx]
    )
    
##HERE WE JOIN ALL RETRIEVED CHUNKS TO THE CONTEXT OUTPUT
##PROMPT GROUNDING
##GIVE LLM CONTEXT TO AVOID HALUCINATION
    context="\n".join(retrieved_chunks)

##LLM CLIENT

    client = Groq(
        api_key="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"
    
    )

    prompt = f"""
    You are a practical AI engineering assistant.

    Answer naturally and conversationally.

    Use provided context for factual grounding.

    Do not give repetitive appreciation,
    motivation, or congratulations.

    Only appreciate when truly appropriate.

    Focus on concise, technically useful answers.

    Context:
    {context}

    Question:
    {query}
    """

    response = client.chat.completions.create(

        model="llama-3.1-8b-instant",

        temperature=0.2,

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    answer = response.choices[0].message.content

    print(answer)




    
    
 

