from abc import ABC, abstractmethod


class TranscriptRetriever(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def query(self, dialogue: str) -> list[int]:
        """
        Return a list of page indices of the retrieved transcripts.
        """
        pass

    @abstractmethod
    def book_name(self) -> str:
        """
        Which book is this retriever looking at?
        """
        pass


class SceneRetriever(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def query(self, scene_description: str) -> list[int]:
        """
        Return a list of page indices of the retrieved scenes.
        """
        pass

    @abstractmethod
    def book_name(self) -> str:
        """
        Which book is this retriever looking at?
        """
        pass