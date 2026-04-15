# Full Duplex Bench v2 — Pipeline Overview

Full Duplex Bench v2 (FDB v2) is an automated benchmark for evaluating spoken, real-time dialogue systems. It measures how well a voice AI model handles natural, full-duplex conversation — including turn-taking, interruption handling, multi-turn instruction following, and task-specific behaviors — without any human involvement in the loop.

The pipeline runs in five sequential stages: generating conversations, preparing the raw outputs for evaluation, transcribing the audio, judging the transcripts with an LLM, and aggregating the scores.

---

## Stage 1 — Conversation Generation

### What this stage is about
Two voice AI agents are placed into a live, real-time spoken conversation. One plays the role of the **Examiner** — it drives the interaction, introduces tasks progressively, and probes the other agent's capabilities. The other plays the **Examinee** — it must respond naturally, follow instructions across turns, and handle interruptions. The entire conversation is a real-time audio exchange; no text is passed between agents at runtime.

### Inputs
- A dataset of task prompts, each specifying what the Examiner should accomplish and how it should reveal goals to the Examinee across the conversation (staged as T1 through T4)
- A system prompt for each role
- Configuration for voice activity detection (VAD) behavior — how quickly each side responds or yields the floor

### Processing
- A signaling hub manages the real-time audio exchange between the two agents, forwarding audio from each side to the other at a strict low-latency cadence
- The Examiner is initialized with a task prompt and begins the conversation; the Examinee is held silent briefly at the start to let the Examiner speak first
- Both sides speak and listen simultaneously (full-duplex), with VAD controlling when each agent responds or yields
- The entire conversation is recorded — each agent's audio is captured separately on its own channel
- After a fixed duration (120 seconds), the conversation is terminated and the two audio channels are mixed into a single stereo recording

### Outputs
- A mono audio recording for each agent (Examiner channel, Examinee channel)
- A combined stereo recording (Examiner on left, Examinee on right)

### Summary
A fully automated spoken conversation is generated between an AI Examiner and an AI Examinee, recorded to audio, with no human intervention.

---

## Stage 2 — Output Preparation

### What this stage is about
The raw audio outputs from Stage 1 are reorganized, annotated, and trimmed into a clean, consistent structure so they are ready for transcription and evaluation.

### Inputs
- Raw audio recordings from Stage 1, potentially nested within session subdirectories
- The original task prompt dataset (to recover task metadata)

### Processing
- Audio files are moved out of any session-scoped subdirectories into a flat, per-task directory layout
- Task metadata — including the system prompt, task prompt, and the staged T1–T4 goals — is written alongside each conversation's audio
- All audio is trimmed to a consistent maximum duration (120 seconds) to ensure uniform evaluation windows

### Outputs
- Per-task directories, each containing the Examiner audio, Examinee audio, combined audio, and a metadata file with the task prompt and staged goals

### Summary
Raw outputs are cleaned up, annotated with task context, and trimmed to a uniform length before any analysis begins.

---

## Stage 3 — ASR Transcription

### What this stage is about
The audio recordings for each conversation are converted into time-stamped text transcripts, one per agent channel.

### Inputs
- The Examiner and Examinee mono audio recordings for each task (from Stage 2)

### Processing
- An automatic speech recognition (ASR) model is run on each audio channel independently
- The output captures both the full continuous transcript text and word-level timestamps — when each word was spoken within the recording

### Outputs
- Two transcript files per conversation (one per agent channel), each containing:
  - The full transcript as a single string
  - A sequence of timestamped word or phrase chunks with start and end times

### Summary
Audio is converted to time-aligned text transcripts, enabling the next stage to reason about what was said and when.

---

## Stage 4 — LLM Evaluation

### What this stage is about
An LLM judge reads the transcripts from both agents and scores the Examinee's performance across multiple dimensions: how naturally it managed turn-taking, how well it followed multi-turn instructions, and how it performed on the task-specific goals defined in the prompt.

### Inputs
- The Examiner and Examinee transcripts with timestamps (from Stage 3)
- The staged task goals (T1–T4) for the conversation
- Evaluation rubrics covering turn-taking fluency, multi-turn instruction following, and task-specific criteria (which vary by task category: Daily, Correction, Entity Tracking, Safety)

### Processing
- A single consolidated prompt is constructed, containing the rubrics, the staged goals, and both transcripts (with the Examinee's transcript including word-level timestamps)
- The judge LLM evaluates the conversation and produces scores per turn-taking event (using timestamps to identify discrete exchange windows) as well as an overall task-specific score
- The judge operates at low temperature for consistent, near-deterministic scoring; failed requests are retried with exponential backoff

### Outputs
- Per-task evaluation results, each containing:
  - A list of turn-taking events, each with a time window and two scores: turn-taking fluency (1–5) and instruction-following quality (1–5)
  - A single task-specific score (1–5)

### Summary
An LLM judge reads both sides of the conversation and produces structured, timestamped scores across turn-taking, instruction-following, and task-specific rubrics.

---

## Stage 5 — Scoring and Aggregation

### What this stage is about
Individual per-task evaluation results are aggregated into summary statistics, enabling comparison across tasks, task categories, and time windows within a conversation.

### Inputs
- All per-task evaluation result files from Stage 4

### Processing
- Scores are aggregated across tasks within each split (Daily, Correction, Entity Tracking, Safety)
- Time-binned aggregation breaks each 120-second conversation into 15-second windows and computes average scores per window — capturing how performance evolves over the course of a conversation
- Per-task-category summaries are computed separately for the task-specific scores

### Outputs
- An overall summary of average scores per split
- A time-binned summary (15-second intervals) showing score trends across the conversation
- A per-task-category breakdown of task-specific scores

### Summary
Per-task scores are rolled up into split-level and time-binned summaries, providing both a top-level benchmark number and insight into how model behavior changes across the arc of a conversation.
