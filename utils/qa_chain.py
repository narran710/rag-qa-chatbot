from groq import Groq

def create_qa_chain():
    client = Groq()

    def ask_question(context, question):
        prompt = f"""
        Answer the question based on the context below.

        Context:
        {context}

        Question:
        {question}

        Answer:
        """

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        answer = response.choices[0].message.content

        return {
            "result": answer,
            "confidence": "High (API-based model)",
            "source_documents": context[:500]
        }

    return ask_question