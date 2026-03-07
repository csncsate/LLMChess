import chess
import random
import re
import torch
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from chess_tournament.players import Player

class TransformerPlayer(Player):
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
        if self.model is None:
            print(f"[{self.name}] Loading {self.model_id} on {self.device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16,
                device_map="auto"
                )
            self.model.eval()

    # -------------------------
    # Prompt
    # -------------------------
    def _build_prompt(self, fen: str) -> str:
        epd = " ".join(fen.split()[:4])
        return f"FEN: {epd}\nMove:"
    
    def _score_moves_batch(self, prompt: str, moves: list) -> list:        
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
    
    def _is_hanging(self, board: chess.Board, move: chess.Move) -> bool:
        piece = board.piece_at(move.from_square)
        if piece is None:
            return False

        board.push(move)
        to_sq = move.to_square

        # is the square attacked by opponent?
        opponent = board.turn  # turn flipped after push
        if board.is_attacked_by(opponent, to_sq):
            me = not opponent
            if not board.is_attacked_by(me, to_sq):
                board.pop()
                return True  # hanging - attacked and undefended

        board.pop()
        return False
    
    def _heuristic_bonus(self, fen: str, move: str) -> float:
        board = chess.Board(fen)
        chess_move = chess.Move.from_uci(move)
        bonus = 0.0
    
        # checkmate
        board.push(chess_move)
        if board.is_checkmate():
            return 1000.0
        if board.is_check():
            bonus += 2.0
        board.pop()
    
        # capture
        if board.is_capture(chess_move):
            bonus += 1.0
    
        # penalize hanging pieces
        if self._is_hanging(board, chess_move):
            bonus -= 3.0
    
        return bonus
    
        return bonus

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
            scores = [s + self._heuristic_bonus(fen, m) for s, m in zip(scores, legal_moves)]
            return legal_moves[scores.index(max(scores))]
        except Exception:
            return self._random_legal(fen)