import chess
import random
import re
import torch
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM

from chess_tournament.players import Player

_MODEL_CACHE = {}

class TransformerPlayer(Player):
    """    
    REQUIRED:
        Subclasses chess_tournament.players.Player
    """

    UCI_REGEX = re.compile(r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b", re.IGNORECASE)

    def __init__(
        self,
        name: str = "LLMChess",
        model_id: str = "csncsate/LLMChess",
    ):
        super().__init__(name)
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None

    # -------------------------
    # Lazy loading
    # -------------------------
    def _load_model(self):
        global _MODEL_CACHE
        if "model" not in _MODEL_CACHE:
            print(f"[{self.name}] Loading {self.model_id} on {self.device}...")
            tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            model.eval()
            _MODEL_CACHE["model"] = model
            _MODEL_CACHE["tokenizer"] = tokenizer
        self.model = _MODEL_CACHE["model"]
        self.tokenizer = _MODEL_CACHE["tokenizer"]

    # -------------------------
    # Prompt
    # -------------------------
    def _build_prompt(self, fen: str) -> str:
        return f"FEN: {fen}\nMove:"
    
    def _score_moves_batch(self, prompt: str, moves: list) -> list:
        prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        prompt_len = prompt_ids.shape[1]

        fulls = [prompt + " " + m for m in moves]
        inputs = self.tokenizer(fulls, return_tensors="pt", padding=True).to(self.model.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits

        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        scores = []
        for i, move in enumerate(moves):
            ids = inputs.input_ids[i]
            move_ids = self.tokenizer(" " + move, return_tensors="pt").input_ids[0][1:]
            move_len = len(move_ids)
            end = ids.tolist().index(self.tokenizer.pad_token_id) if self.tokenizer.pad_token_id in ids.tolist() else len(ids)
            start = end - move_len
            score = sum(log_probs[i, start + j - 1, ids[start + j]].item() for j in range(move_len))
            scores.append(score)
        return scores

    def _random_legal(self, fen: str) -> Optional[str]:
        board = chess.Board(fen)
        moves = list(board.legal_moves)
        return random.choice(moves).uci() if moves else None

    # -------------------------
    # Main API
    # -------------------------
    def get_move(self, fen: str) -> Optional[str]:
        try:
            self._load_model()
        except Exception:
            return self._random_legal(fen)
    
        board = chess.Board(fen)
        legal_moves = [m.uci() for m in board.legal_moves]
        if not legal_moves:
            return None
    
        prompt = self._build_prompt(fen)
    
        try:
            scores = self._score_moves_batch(prompt, legal_moves)
            return legal_moves[scores.index(max(scores))]
        except Exception:
            return self._random_legal(fen)