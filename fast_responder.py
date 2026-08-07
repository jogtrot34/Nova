"""
fast_responder.py

*** STANDALONE — not imported by main.py or identify.py. ***
*** Not wired into the running system yet, on purpose.   ***

This is your existing bag-of-words intent-classifier chatbot
(sourcecode.py / annah.py), cleaned up into one file, with two
concrete changes:

  1. Speaks through Piper (piper_speak.py) instead of edge_tts +
     pygame — consistent with the rest of Nova's voice stack, and
     works fully offline (edge-tts needs internet; Piper doesn't).
  2. Uses relative paths (intents.json, chatbot_model.pth,
     dimensions.json — all relative to wherever you run this from)
     instead of anything hardcoded.

The idea, as I understand it: this is a *fast lane* for small talk /
simple commands — a tiny model that answers in milliseconds — sitting
alongside Qwen (brain.py) for anything that actually needs real
reasoning. voice_layer.py keeps answering "who is speaking"; this
answers "what did they say, and what's the canned reply." Both would
run on the same short audio clip, just doing different jobs with it.

WHAT'S HERE:
    train_and_save()     — trains the intent classifier from
                            intents.json (you already have real
                            patterns/responses in there)
    respond(text)         — text in, response text out (or None if no
                            intent matched with confidence)
    listen_and_respond()  — one full loop: mic -> Google STT -> intent
                            match -> Piper speaks the reply

WHAT THIS DOESN'T DO YET (deliberately, since you said not to wire it
in): it doesn't listen to Nova's live camera/mic loop, doesn't get
called from main.py, and doesn't have a way for the web UI to push
audio to it. See the bottom of this file for how that would plug in
when you're ready.

Try it once you've trained a model:
    python3 fast_responder.py --train
    python3 fast_responder.py --say "hello there"
    python3 fast_responder.py --listen        # one mic round-trip
"""

import argparse
import json
import os
import random

import nltk
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

INTENTS_PATH    = "intents.json"
MODEL_PATH      = "chatbot_model.pth"
DIMENSIONS_PATH = "dimensions.json"

# Below this, don't trust the match — better to say nothing (and let
# Qwen or a plain "I didn't catch that" handle it) than to confidently
# answer the wrong intent.
CONFIDENCE_FLOOR = 0.55


class ChatbotModel(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)


class FastResponder:
    def __init__(self, intents_path: str = INTENTS_PATH):
        self.intents_path = intents_path
        self.model = None
        self.documents = []
        self.vocabulary = []
        self.intents = []
        self.intents_responses = {}

    @staticmethod
    def _tokenize_and_lemmatize(text):
        lemmatizer = nltk.WordNetLemmatizer()
        words = nltk.word_tokenize(text)
        return [lemmatizer.lemmatize(w.lower()) for w in words]

    def _bag_of_words(self, words):
        return [1 if w in words else 0 for w in self.vocabulary]

    def _parse_intents(self):
        with open(self.intents_path, "r") as f:
            data = json.load(f)

        for intent in data["intents"]:
            if intent["tag"] not in self.intents:
                self.intents.append(intent["tag"])
                self.intents_responses[intent["tag"]] = intent["responses"]
            for pattern in intent["patterns"]:
                words = self._tokenize_and_lemmatize(pattern)
                self.vocabulary.extend(words)
                self.documents.append((words, intent["tag"]))

        self.vocabulary = sorted(set(self.vocabulary))

    def train_and_save(self, batch_size=8, lr=0.001, epochs=100,
                       model_path=MODEL_PATH, dimensions_path=DIMENSIONS_PATH):
        self._parse_intents()

        X, y = [], []
        for words, tag in self.documents:
            X.append(self._bag_of_words(words))
            y.append(self.intents.index(tag))
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int64)

        loader = DataLoader(TensorDataset(torch.tensor(X), torch.tensor(y)),
                            batch_size=batch_size, shuffle=True)

        self.model = ChatbotModel(X.shape[1], len(self.intents))
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            running_loss = 0.0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            if (epoch + 1) % 20 == 0 or epoch == epochs - 1:
                print(f"  epoch {epoch+1}/{epochs}  loss={running_loss/len(loader):.4f}")

        torch.save(self.model.state_dict(), model_path)
        with open(dimensions_path, "w") as f:
            json.dump({"input_size": X.shape[1],
                      "output_size": len(self.intents),
                      "vocabulary": self.vocabulary,
                      "intents": self.intents}, f)
        print(f"[FastResponder] Trained on {len(self.documents)} pattern(s), "
              f"{len(self.intents)} intent(s). Saved to {model_path}.")

    def load(self, model_path=MODEL_PATH, dimensions_path=DIMENSIONS_PATH):
        with open(dimensions_path, "r") as f:
            dims = json.load(f)
        self.vocabulary = dims["vocabulary"]
        self.intents    = dims["intents"]

        with open(self.intents_path, "r") as f:
            data = json.load(f)
        for intent in data["intents"]:
            self.intents_responses[intent["tag"]] = intent["responses"]

        self.model = ChatbotModel(dims["input_size"], dims["output_size"])
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model.eval()
        print(f"[FastResponder] Loaded — {len(self.intents)} intent(s).")

    def respond(self, text: str):
        """Returns a response string, or None if nothing matched
        confidently enough to trust (caller should fall back to Qwen
        or a generic 'could you repeat that')."""
        if self.model is None:
            raise RuntimeError("Call load() or train_and_save() first.")

        words = self._tokenize_and_lemmatize(text)
        bag = torch.tensor([self._bag_of_words(words)], dtype=torch.float32)

        with torch.no_grad():
            logits = self.model(bag)
            probs  = torch.softmax(logits, dim=1)[0]
        best_idx  = int(torch.argmax(probs).item())
        best_prob = float(probs[best_idx])

        if best_prob < CONFIDENCE_FLOOR:
            return None

        tag = self.intents[best_idx]
        responses = self.intents_responses.get(tag)
        return random.choice(responses) if responses else None


# ── Optional full loop: mic -> STT -> respond -> Piper ────────────────────

def listen_once() -> str | None:
    """Google's speech_recognition backend — needs internet. Swap for
    an offline STT (Vosk, whisper.cpp) later if you want this fully
    offline like the rest of Nova."""
    import speech_recognition as sr
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return None
    except sr.RequestError as e:
        print(f"[FastResponder] STT request failed (no internet?): {e}")
        return None


def listen_and_respond(responder: "FastResponder"):
    from piper_speak import speak
    text = listen_once()
    if not text:
        print("[FastResponder] Didn't catch that.")
        return
    print(f"Heard: {text!r}")
    reply = responder.respond(text)
    if reply is None:
        print("[FastResponder] No confident match — this is where you'd "
              "hand off to Qwen instead.")
        return
    print(f"Replying: {reply!r}")
    speak(reply)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true",
                        help="Train from intents.json and save the model")
    parser.add_argument("--say", type=str,
                        help="Text in, response text out (no mic/TTS)")
    parser.add_argument("--listen", action="store_true",
                        help="One full mic -> STT -> reply -> Piper round-trip")
    args = parser.parse_args()

    responder = FastResponder()

    if args.train:
        responder.train_and_save()
    elif args.say:
        responder.load()
        print(responder.respond(args.say))
    elif args.listen:
        responder.load()
        listen_and_respond(responder)
    else:
        parser.print_help()
