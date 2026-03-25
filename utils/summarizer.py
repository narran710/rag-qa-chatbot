from transformers import pipeline


summarizer = pipeline("summarization")


def summarize_document(text):

    summary = summarizer(
        text,
        max_length=150,
        min_length=40,
        do_sample=False
    )

    return summary[0]['summary_text']