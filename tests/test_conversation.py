from src.rag.conversation import ConversationMemoryStore, ConversationTurn, build_contextual_query


def test_conversation_memory_is_scoped_and_bounded():
    store = ConversationMemoryStore(max_turns=2)

    store.append("s1", "w1", "public", ConversationTurn(user="first", assistant="a1"))
    store.append("s1", "w1", "public", ConversationTurn(user="second", assistant="a2"))
    store.append("s1", "w1", "public", ConversationTurn(user="third", assistant="a3"))
    store.append("s2", "w1", "public", ConversationTurn(user="other", assistant="a4"))

    turns = store.get("s1", "w1", "public")

    assert [turn.user for turn in turns] == ["second", "third"]
    assert [turn.user for turn in store.get("s2", "w1", "public")] == ["other"]


def test_contextual_query_only_rewrites_follow_ups():
    turns = [ConversationTurn(user="What evidence is required for vendor onboarding?", assistant="SOC 2. [a]")]

    assert build_contextual_query("What about its renewal?", turns) == (
        "What evidence is required for vendor onboarding? What about its renewal?"
    )
    assert build_contextual_query("Explain payroll controls", turns) == "Explain payroll controls"
