
import tempfile
from pathlib import Path

from backend.conversation import ConversationManager


def test_persistent_conversation():

    with tempfile.TemporaryDirectory() as temp_dir:

        db_path = Path(temp_dir) / "test.db"

        manager = ConversationManager(
            db_path=db_path
        )

        session_id = "test-session-001"

        manager.create_session(session_id)

        manager.add_message(
            session_id,
            "user",
            "What are the risks?"
        )

        manager.add_message(
            session_id,
            "assistant",
            "The document discusses security risks.",
            metadata={
                "citations": [
                    {
                        "reference": "[Source 1]",
                        "page": 4
                    }
                ]
            }
        )

        # Verify history.
        history = manager.get_history(session_id)

        assert len(history) == 2

        # Verify complete conversation.
        conversation = manager.get_session(session_id)

        assert conversation is not None
        assert conversation["title"] == "What are the risks?"
        assert len(conversation["messages"]) == 2

        # Verify citation persistence.
        metadata = conversation["messages"][1]["metadata"]

        assert metadata["citations"][0]["page"] == 4

        # Simulate application restart.
        manager_after_restart = ConversationManager(
            db_path=db_path
        )

        restored = manager_after_restart.get_session(
            session_id
        )

        assert restored is not None
        assert len(restored["messages"]) == 2

        # Verify rename.
        assert manager_after_restart.rename_session(
            session_id,
            "Risk Analysis"
        )

        assert (
            manager_after_restart.get_session(session_id)["title"]
            == "Risk Analysis"
        )

        # Verify deletion.
        assert manager_after_restart.delete_session(
            session_id
        )

        assert manager_after_restart.get_session(
            session_id
        ) is None

        print("All conversation persistence tests passed!")


if __name__ == "__main__":
    test_persistent_conversation()
