import asyncio
import numpy as np
from pydub.audio_segment import AudioSegment


def load_keyboard_sound(
    path: str, target_rate: int = 16000, chunk_ms: int = 100
) -> list[tuple[int, np.ndarray]]:
    audio = AudioSegment.from_file(path).set_channels(1).set_frame_rate(target_rate)
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    samples /= 32768.0

    samples_per_chunk = int(target_rate * chunk_ms / 1000)
    chunks = []
    for i in range(0, len(samples), samples_per_chunk):
        chunk = samples[i : i + samples_per_chunk]
        if len(chunk) == 0:
            continue
        chunks.append((target_rate, chunk))

    return chunks


KEYBOARD_AUDIO_CHUNKS = load_keyboard_sound("sounds/keyboard.mp3")


async def stream_keyboard_sound(max_duration_s: float = 3.0):
    """
    Streams a random keyboard sound effect for at most `max_duration_s` seconds.

    Args:
        max_duration_s: Maximum duration (in seconds) to stream.
    """
    if max_duration_s <= 0:
        return

    total_samples = 0
    total_samples_allowed: int | None = None

    for sample_rate, chunk in KEYBOARD_AUDIO_CHUNKS:
        # Initialize allowed sample budget once we know the sample rate
        if total_samples_allowed is None:
            total_samples_allowed = int(max_duration_s * sample_rate)

        if total_samples >= total_samples_allowed:
            break

        remaining_samples = total_samples_allowed - total_samples

        # Trim the chunk if it would exceed the allowed duration
        if len(chunk) > remaining_samples:
            chunk = chunk[:remaining_samples]

        if len(chunk) == 0:
            break

        yield (sample_rate, chunk)
        total_samples += len(chunk)

        await asyncio.sleep(0)  # allow event loop to breathe
