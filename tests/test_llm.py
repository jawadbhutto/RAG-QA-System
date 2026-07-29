import src.llm as llm_module


def test_parse_model_string_splits_on_first_colon_only():
    provider, model = llm_module._parse_model_string("ollama:qwen2.5:3b")
    assert provider == "ollama"
    assert model == "qwen2.5:3b"  # tag's own colon is preserved


def test_parse_model_string_openai():
    provider, model = llm_module._parse_model_string("openai:gpt-4o-mini")
    assert provider == "openai"
    assert model == "gpt-4o-mini"


def test_parse_model_string_without_colon_raises():
    import pytest

    with pytest.raises(ValueError):
        llm_module._parse_model_string("justmodelname")


def test_get_llm_passes_num_predict_and_base_url_for_ollama(monkeypatch):
    captured = {}

    def fake_init_chat_model(model_name, model_provider=None, **kwargs):
        captured["model_name"] = model_name
        captured["model_provider"] = model_provider
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(llm_module.settings, "llm_model", "ollama:qwen2.5:3b")
    monkeypatch.setattr(llm_module.settings, "llm_temperature", 0.0)
    monkeypatch.setattr(llm_module.settings, "llm_max_tokens", 800)
    monkeypatch.setattr(llm_module.settings, "ollama_base_url", "http://localhost:11434")

    import langchain.chat_models as chat_models_module
    monkeypatch.setattr(chat_models_module, "init_chat_model", fake_init_chat_model)

    llm_module.get_llm.cache_clear()
    llm_module.get_llm()

    assert captured["model_name"] == "qwen2.5:3b"
    assert captured["model_provider"] == "ollama"
    assert captured["kwargs"]["num_predict"] == 800
    assert "max_tokens" not in captured["kwargs"]
    assert captured["kwargs"]["base_url"] == "http://localhost:11434"


def test_get_llm_uses_max_tokens_for_openai(monkeypatch):
    captured = {}

    def fake_init_chat_model(model_name, model_provider=None, **kwargs):
        captured["model_name"] = model_name
        captured["model_provider"] = model_provider
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(llm_module.settings, "llm_model", "openai:gpt-4o-mini")
    monkeypatch.setattr(llm_module.settings, "llm_temperature", 0.0)
    monkeypatch.setattr(llm_module.settings, "llm_max_tokens", 800)

    import langchain.chat_models as chat_models_module
    monkeypatch.setattr(chat_models_module, "init_chat_model", fake_init_chat_model)

    llm_module.get_llm.cache_clear()
    llm_module.get_llm()

    assert captured["model_provider"] == "openai"
    assert captured["kwargs"]["max_tokens"] == 800
    assert "num_predict" not in captured["kwargs"]
