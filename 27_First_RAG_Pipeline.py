from haystack import Pipeline
from haystack.components.generators import HuggingFaceLocalGenerator
from haystack.components.builders import PromptBuilder
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever, InMemoryEmbeddingRetriever
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.preprocessors.document_splitter import DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.components.joiners import DocumentJoiner

document_store = InMemoryDocumentStore()


docs = [
    Document(
        content="ایران یک کشور چهار فصل و دارای فرهنگی غنی است که پایتخت آن تهران است."),
    Document(content="اصفهان یکی از شهرهای تاریخی ایران است."),
    Document(content="زبان رسمی ایران فارسی است."),
    Document(content="ایران دارای تاریخ و فرهنگ غنی است.")
]

doc_splitter = DocumentSplitter(
    split_by="word", split_length=512, split_overlap=32)
doc_embedder = SentenceTransformersDocumentEmbedder(
    model="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

doc_writer = DocumentWriter(document_store)

indexing_pipeline = Pipeline()
indexing_pipeline.add_component("doc_splitter", doc_splitter)
indexing_pipeline.add_component("doc_embedder", doc_embedder)
indexing_pipeline.add_component("doc_writer", doc_writer)

indexing_pipeline.connect("doc_splitter", "doc_embedder")
indexing_pipeline.connect("doc_embedder", "doc_writer")

indexing_pipeline.run({"doc_splitter": {"documents": docs}})


document_joiner = DocumentJoiner(join_mode='merge')


text_embedder = SentenceTransformersTextEmbedder(
    model="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
text_embedder.warm_up()


retriever = InMemoryEmbeddingRetriever(document_store)
bm25_retriever = InMemoryBM25Retriever(document_store)


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
    task="text2text-generation",
    generation_kwargs={"max_new_tokens": 100})
generator.warm_up()


basic_rag_pipeline = Pipeline()
# Add components to your pipeline
basic_rag_pipeline.add_component("text_embedder", text_embedder)
basic_rag_pipeline.add_component("retriever", retriever)
basic_rag_pipeline.add_component("bm25_retriever", bm25_retriever)
basic_rag_pipeline.add_component("document_joiner", document_joiner)
basic_rag_pipeline.add_component("prompt_builder", prompt_builder)
basic_rag_pipeline.add_component("llm", generator)

# Now, connect the components to each other
basic_rag_pipeline.connect("text_embedder", "retriever.query_embedding")
basic_rag_pipeline.connect("retriever", "document_joiner")
basic_rag_pipeline.connect("bm25_retriever", "document_joiner")
basic_rag_pipeline.connect("document_joiner", "prompt_builder.documents")
basic_rag_pipeline.connect("prompt_builder", "llm")


question = "پایتخت ایران کجاست؟"

retrieve_doc = text_embedder.run(question)['embedding']
retriever_test = retriever.run(retrieve_doc)
print("Doc:", retriever_test['documents'])

response = basic_rag_pipeline.run(
    {
        "text_embedder": {"text": question},
        "retriever": {"top_k": 3},
        "bm25_retriever": {"query": question, "top_k": 3},
        "document_joiner": {"top_k": 5},
        "prompt_builder": {"question": question}
    })
print('Answer:', response["llm"]["replies"][0])
