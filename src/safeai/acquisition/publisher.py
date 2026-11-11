"""Publication en mémoire vers les modules suivants — files bornées, non bloquantes."""

from __future__ import annotations

import queue
from typing import TypeVar

from safeai.acquisition.config import PublisherSettings, QueueSettings
from safeai.acquisition.contracts import AudioChunk, AudioQualityEvent, SpeechSegment

T = TypeVar("T")


class BoundedFanout:
    """Diffuse un événement vers plusieurs abonnés sans bloquer la capture."""

    def __init__(self, maxsize: int, drop_oldest: bool = True) -> None:
        self.maxsize = maxsize
        self.drop_oldest = drop_oldest
        self._subscribers: list[queue.Queue[T]] = []
        self.dropped = 0

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)

    def subscribe(self) -> queue.Queue[T]:
        subscriber: queue.Queue[T] = queue.Queue(maxsize=self.maxsize)
        self._subscribers.append(subscriber)
        return subscriber

    def publish(self, item: T) -> None:
        for subscriber in list(self._subscribers):
            try:
                subscriber.put_nowait(item)
            except queue.Full:
                self.dropped += 1
                if not self.drop_oldest:
                    continue
                try:
                    subscriber.get_nowait()
                except queue.Empty:
                    pass
                try:
                    subscriber.put_nowait(item)
                except queue.Full:
                    pass

    def drain(self) -> None:
        """Vide les files et remet le compteur à zéro, sans retirer les abonnés.

        Une nouvelle session ne doit pas effacer le consommateur qui vient
        d'être branché pour la recevoir.
        """
        for subscriber in self._subscribers:
            while True:
                try:
                    subscriber.get_nowait()
                except queue.Empty:
                    break
        self.dropped = 0


class AcquisitionPublisher:
    def __init__(
        self,
        queues: QueueSettings | None = None,
        publisher: PublisherSettings | None = None,
    ) -> None:
        queue_settings = queues or QueueSettings()
        pub_settings = publisher or PublisherSettings()
        self.audio = BoundedFanout(queue_settings.audio_maxsize, pub_settings.drop_oldest_on_full)
        self.speech = BoundedFanout(queue_settings.speech_maxsize, pub_settings.drop_oldest_on_full)
        self.quality = BoundedFanout(queue_settings.quality_maxsize, pub_settings.drop_oldest_on_full)

    def subscribe_audio(self) -> queue.Queue[AudioChunk]:
        return self.audio.subscribe()

    def subscribe_speech_segments(self) -> queue.Queue[SpeechSegment]:
        return self.speech.subscribe()

    def subscribe_quality_events(self) -> queue.Queue[AudioQualityEvent]:
        return self.quality.subscribe()

    def publish_audio(self, chunk: AudioChunk) -> None:
        chunk.validate()
        self.audio.publish(chunk)

    def publish_speech(self, segment: SpeechSegment) -> None:
        segment.validate()
        self.speech.publish(segment)

    def publish_quality(self, event: AudioQualityEvent) -> None:
        self.quality.publish(event)

    @property
    def dropped_total(self) -> int:
        return self.audio.dropped + self.speech.dropped + self.quality.dropped

    def reset(self) -> None:
        self.audio.drain()
        self.speech.drain()
        self.quality.drain()
