import asyncio
import aiohttp
import soundfile as sf
import numpy as np
import sphn
import os

# Configuration - adjust to your server's settings
URL = "http://localhost:8998/api/chat"
INPUT_AUDIO = "/fs/gamma-projects/audio/raman/steerd/personaplex/assets/test/input_assistant.wav"  # Your hardcoded input variable
OUTPUT_FILE = "output.wav"
SAMPLE_RATE = 24000        # Standard Moshi sample rate

async def run_test():
    # Load and preprocess input audio
    if not os.path.exists(INPUT_AUDIO):
        print(f"Error: {INPUT_AUDIO} not found.")
        return

    audio_data, sr = sf.read(INPUT_AUDIO)
    if sr != SAMPLE_RATE:
        # In a real setup, you'd resample here. 
        # For this test, we assume the input matches the model.
        print(f"Warning: Input SR {sr} doesn't match {SAMPLE_RATE}")
    
    if audio_data.ndim > 1:
        audio_data = audio_data[:, 0]  # Convert to mono

    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(URL) as ws:
            print(f"Connected to {URL}")

            # 1. Wait for Handshake (0x00)
            handshake = await ws.receive_bytes()
            if handshake == b"\x00":
                print("Handshake successful: Server is ready.")

            # Audio state
            writer = sphn.OpusStreamWriter(SAMPLE_RATE)
            reader = sphn.OpusStreamReader(SAMPLE_RATE)
            output_chunks = []

            async def send_loop():
                # Send audio in 80ms chunks (1920 samples @ 24kHz)
                chunk_size = 1920 
                for i in range(0, len(audio_data), chunk_size):
                    chunk = audio_data[i:i + chunk_size].astype(np.float32)
                    
                    # Encode and send
                    opus_bytes = writer.append_pcm(chunk)
                    if opus_bytes:
                        await ws.send_bytes(b"\x01" + opus_bytes)
                    
                    # Simulate real-time streaming
                    await asyncio.sleep(0.08) 
                
                print("\n[Client] Finished sending input audio.")

            async def recv_loop():
                print("[Client] Listening for response...")
                try:
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.BINARY:
                            kind = msg.data[0]
                            payload = msg.data[1:]

                            if kind == 1:  # Audio Kind
                                pcm = reader.append_bytes(payload)
                                if pcm.size > 0:
                                    output_chunks.append(pcm)
                                    # Log buffering status
                                    print(f"--- Buffering: {len(output_chunks) * 80}ms received", end="\r")
                            
                            elif kind == 2:  # Text Kind
                                print(f"\n[Model Text]: {payload.decode('utf-8')}")
                except Exception as e:
                    print(f"\nStream ended: {e}")

            # Run sender and receiver concurrently
            await asyncio.gather(send_loop(), recv_loop())

            # 2. Save Response
            if output_chunks:
                full_audio = np.concatenate(output_chunks)
                sf.write(OUTPUT_FILE, full_audio, SAMPLE_RATE)
                print(f"\n[Success] Response saved to {OUTPUT_FILE}")
            else:
                print("\n[Error] No audio response received.")

if __name__ == "__main__":
    asyncio.run(run_test())