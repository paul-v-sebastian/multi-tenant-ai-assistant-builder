from app import is_chat_ready


def test_is_chat_ready_requires_loaded_document():
    assert is_chat_ready({"namespace": "ns-123", "document_name": "doc.pdf", "chunk_count": 4}) is True
    assert is_chat_ready({"namespace": None, "document_name": "doc.pdf", "chunk_count": 4}) is False
    assert is_chat_ready({"namespace": "ns-123", "document_name": None, "chunk_count": 4}) is False
    assert is_chat_ready({"namespace": "ns-123", "document_name": "doc.pdf", "chunk_count": 0}) is False
