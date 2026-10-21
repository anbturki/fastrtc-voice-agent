import asyncio
import sys
import time
from config import settings
from langchain_groq.chat_models import ChatGroq
from fastrtc.text_to_speech.tts import get_tts_model
from fastrtc.speech_to_text.stt_ import get_stt_model
from fastrtc import Stream, ReplyOnPause
import numpy as np
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from loguru import logger
from langchain.tools import tool
import gradio as gr
from sounds.load import stream_keyboard_sound

# Configure loguru for human-readable output
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="DEBUG",
    colorize=True,
)

logger.info("=" * 60)
logger.info("          VOICE AGENT - STARTING UP")
logger.info("=" * 60)

system_prompt = """
Your name is Lisa, and you work for The Neural Maze real estate company.
Your task is to provide information about specific apartments using the `search_property_mock_tool`.
Don't use asterisks or emojis, as you are engaged in a phone call. Just return short and informative responses.
"""

logger.debug(f"System prompt loaded ({len(system_prompt)} chars)")


@tool
def search_property_mock_tool(location: str) -> str:
    """Retrieve real estate details for properties in a given location."""
    logger.info("[TOOL CALL] search_property_mock_tool invoked")
    logger.debug(f"  -> Location query: '{location}'")

    result = (
        "I found one apartment in that area. It features 3 rooms, "
        "2 bathrooms, and a beautifully designed living room."
    )

    logger.debug(f"  <- Tool response: '{result[:50]}...'")
    return result


logger.info("Loading STT model...")
stt_start = time.time()
stt_model = get_stt_model()
logger.success(f"STT model loaded in {time.time() - stt_start:.2f}s")

logger.info(f"Initializing LLM: {settings.groq.model}")
llm = ChatGroq(model=settings.groq.model, api_key=settings.groq.api_key)
logger.success("LLM initialized")

logger.info("Loading TTS model...")
tts_start = time.time()
tts_model = get_tts_model()
logger.success(f"TTS model loaded in {time.time() - tts_start:.2f}s")

logger.info("Creating agent with tools: [search_property_mock_tool]")
simple_agent = create_agent(
    llm,
    checkpointer=InMemorySaver(),
    system_prompt=system_prompt,
    tools=[search_property_mock_tool],
)
logger.success("Agent created successfully")


def model_has_tool_calls(model_step_data) -> bool:
    logger.trace("Checking for tool calls in model response...")
    msgs = None
    if isinstance(model_step_data, dict) and "messages" in model_step_data:
        msgs = model_step_data["messages"]
        logger.trace(f"  Found {len(msgs)} messages in dict format")
    elif isinstance(model_step_data, list):
        msgs = model_step_data
        logger.trace(f"  Found {len(msgs)} messages in list format")
    else:
        msgs = [model_step_data]
        logger.trace("  Single message format")

    for i, msg in enumerate(msgs):
        if getattr(msg, "tool_calls", None):
            logger.debug(f"  Message {i}: has tool_calls attribute")
            return True

        if isinstance(msg, dict) and msg.get("tool_calls"):
            logger.debug(f"  Message {i}: has tool_calls in dict")
            return True

        content = getattr(msg, "content", None) or (
            msg.get("content") if isinstance(msg, dict) else None
        )
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("tool_calls"):
                    logger.debug(f"  Message {i}: has tool_calls in content list")
                    return True

    logger.trace("  No tool calls found")
    return False


# Request counter for tracking
request_counter = 0


async def simple_agent_handler(audio: tuple[int, np.ndarray], *additional_inputs):
    global request_counter
    request_counter += 1
    request_id = request_counter

    handler_start = time.time()
    logger.info("=" * 50)
    logger.info(f"[REQUEST #{request_id}] New audio received")

    _, tool_use_message, keyboard_sound_seconds = additional_inputs
    sample_rate, audio_data = audio

    logger.debug(f"  Audio info: sample_rate={sample_rate}Hz, shape={audio_data.shape}, dtype={audio_data.dtype}")
    logger.debug(f"  Audio duration: {len(audio_data) / sample_rate:.2f}s")
    logger.debug(f"  Tool use message: '{tool_use_message}'")
    logger.debug(f"  Keyboard sound duration: {keyboard_sound_seconds}s")

    logger.info(f"[REQUEST #{request_id}] Starting speech-to-text...")
    stt_start = time.time()
    transcription = stt_model.stt(audio)
    stt_duration = time.time() - stt_start

    logger.success(f"[REQUEST #{request_id}] Transcription complete in {stt_duration:.2f}s")
    logger.info(f"[REQUEST #{request_id}] User said: \"{transcription}\"")

    if not transcription or transcription.strip() == "":
        logger.warning(f"[REQUEST #{request_id}] Empty transcription, skipping processing")
        return

    logger.info(f"[REQUEST #{request_id}] Sending to agent...")
    agent_start = time.time()

    final_text = None
    chunk_count = 0

    for chunk in simple_agent.stream(
        {"messages": [{"role": "user", "content": "i'm looking for an apartment in Dubai"}]},
        {"configurable": {"thread_id": "phone_call"}},
    ):
        chunk_count += 1
        for step, data in chunk.items():
            logger.debug(f"[REQUEST #{request_id}] Agent step: '{step}' (chunk #{chunk_count})")

            if step == "model":
                if model_has_tool_calls(data):
                    logger.info(f"[REQUEST #{request_id}] Tool call detected! Playing interim message...")

                    logger.debug(f"[REQUEST #{request_id}] Streaming TTS for tool use message...")
                    tts_chunk_count = 0
                    async for audio_chunk in tts_model.stream_tts(tool_use_message):
                        tts_chunk_count += 1
                        yield audio_chunk
                    logger.debug(f"[REQUEST #{request_id}] Streamed {tts_chunk_count} TTS chunks")

                    logger.debug(f"[REQUEST #{request_id}] Playing keyboard sound for {keyboard_sound_seconds}s...")
                    kb_chunk_count = 0
                    async for kb_chunk in stream_keyboard_sound(keyboard_sound_seconds):
                        kb_chunk_count += 1
                        yield kb_chunk
                    logger.debug(f"[REQUEST #{request_id}] Streamed {kb_chunk_count} keyboard sound chunks")

                    logger.debug(f"[REQUEST #{request_id}] Waiting 5s for tool execution...")
                    await asyncio.sleep(5)
                else:
                    # Extract content from model response
                    if "messages" in data and len(data["messages"]) > 0:
                        msg = data["messages"][0]
                        content = getattr(msg, "content", None) or msg.get("content", "")
                        if content:
                            logger.debug(f"[REQUEST #{request_id}] Model response preview: '{content[:100]}...'")

            elif step == "tools":
                logger.info(f"[REQUEST #{request_id}] Tool execution step")
                if "messages" in data:
                    for msg in data["messages"]:
                        tool_content = getattr(msg, "content", None) or msg.get("content", "")
                        logger.debug(f"[REQUEST #{request_id}] Tool result: '{tool_content[:100]}...'")

            # Update final_text
            if "messages" in data and len(data["messages"]) > 0:
                msg_content = getattr(data["messages"][0], "content", None) or data["messages"][0].get("content", "")
                if msg_content:
                    final_text = msg_content

    agent_duration = time.time() - agent_start
    logger.success(f"[REQUEST #{request_id}] Agent processing complete in {agent_duration:.2f}s ({chunk_count} chunks)")

    if not final_text:
        logger.warning(f"[REQUEST #{request_id}] No final response from agent, using fallback")
        final_text = "I'm sorry, I couldn't find anything useful in the system."

    logger.info(f"[REQUEST #{request_id}] Agent response: \"{final_text}\"")

    logger.info(f"[REQUEST #{request_id}] Starting TTS for final response...")
    tts_start = time.time()
    tts_chunk_count = 0

    async for audio_chunk in tts_model.stream_tts(final_text):
        tts_chunk_count += 1
        if tts_chunk_count == 1:
            logger.debug(f"[REQUEST #{request_id}] First TTS chunk received")
        yield audio_chunk

    tts_duration = time.time() - tts_start
    logger.success(f"[REQUEST #{request_id}] TTS complete: {tts_chunk_count} chunks in {tts_duration:.2f}s")

    total_duration = time.time() - handler_start
    logger.info(f"[REQUEST #{request_id}] Request complete in {total_duration:.2f}s")
    logger.info(f"[REQUEST #{request_id}] Breakdown: STT={stt_duration:.2f}s, Agent={agent_duration:.2f}s, TTS={tts_duration:.2f}s")
    logger.info("=" * 50)


logger.info("Creating FastRTC stream...")
stream = Stream(
    handler=ReplyOnPause(simple_agent_handler),
    modality="audio",
    mode="send-receive",
    additional_inputs=[
        gr.Text("Let me look for that in the system", label="Tool Use Sentence"),
        gr.Number(3.0, label="Max Keyboard Sound Duration"),
    ],
)
logger.success("FastRTC stream created")

logger.info("=" * 60)
logger.info("          VOICE AGENT - READY")
logger.info("=" * 60)
logger.info("Launching Gradio UI...")

stream.ui.launch()
