from haystack import Pipeline
from haystack.components.generators import HuggingFaceLocalGenerator
from haystack.components.builders import PromptBuilder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore

document_store = InMemoryDocumentStore()


docs = [
    Document(content="تهران پایتخت ایران است."),
    Document(content="اصفهان یکی از شهرهای تاریخی ایران است."),
    Document(content="زبان رسمی ایران فارسی است."),
    Document(content="ایران دارای تاریخ و فرهنگ غنی است.")
]

doc_embedder = SentenceTransformersDocumentEmbedder(
    model="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
doc_embedder.warm_up()


docs_with_embeddings = doc_embedder.run(docs)
document_store.write_documents(docs_with_embeddings["documents"])


text_embedder = SentenceTransformersTextEmbedder(
    model="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
text_embedder.warm_up()


retriever = InMemoryEmbeddingRetriever(document_store, top_k=1)


template = """
بر مبنای اطلاعات ارائه شده در ادامه به سوال پاسخ بده.

اطلاعات:
{% for document in documents %}
    {{ document.content }}
{% endfor %}

سوال: {{question}}
پاسخ: 
"""
prompt_builder = PromptBuilder(template=template)


generator = HuggingFaceLocalGenerator(
    model="m3hrdadfi/gpt2-persian-qa",
    task="text2text-generation")
generator.warm_up()


basic_rag_pipeline = Pipeline()
# Add components to your pipeline
basic_rag_pipeline.add_component("text_embedder", text_embedder)
basic_rag_pipeline.add_component("retriever", retriever)
basic_rag_pipeline.add_component("prompt_builder", prompt_builder)
basic_rag_pipeline.add_component("llm", generator)

# Now, connect the components to each other
basic_rag_pipeline.connect("text_embedder.embedding",
                           "retriever.query_embedding")
basic_rag_pipeline.connect("retriever", "prompt_builder.documents")
basic_rag_pipeline.connect("prompt_builder", "llm")


question = "پایتخت ایران کجاست؟"

retrieve_doc = text_embedder.run(question)['embedding']
retriever_test = retriever.run(retrieve_doc)
print("Doc:", retriever_test['documents'])

response = basic_rag_pipeline.run(
    {"text_embedder": {"text": question}, "prompt_builder": {"question": question}})
print('Answer:', response["llm"]["replies"][0])
