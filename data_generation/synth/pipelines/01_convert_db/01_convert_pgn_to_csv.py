import re
import io
import csv
import zstandard as zstd
import pandas as pd
from typing import List, Dict, Tuple, Optional, Iterator
import argparse
from dataclasses import dataclass
from pathlib import Path
from contextlib import contextmanager


@dataclass
class ChessMove:
    """Data class representing a single chess move"""
    move: str
    eval: Optional[float]
    clock: Optional[int]
    half_move: int


@dataclass
class GameMetadata:
    """Data class representing game metadata"""
    game_id: str
    date: str
    white: str
    black: str
    white_elo: int
    black_elo: int
    time_control: str
    termination: str
    result: str


class PGNProcessor:
    """Main processor for converting PGN files to CSV format"""

    # ----------------------------
    # Constants / Regex patterns
    # ----------------------------

    # Keep only games whose [Event "..."] contains this keyword (case-insensitive)
    REQUIRED_EVENT_KEYWORD: str = "blitz"


    # Common patterns for metadata extraction
    # Examples:
    # [Site "https://lichess.org/JRgTjrR2"] -> captures "https://lichess.org/JRgTjrR2"
    # [Date "2023.01.01"] -> captures "2023.01.01"
    # [White "Murzillka"] -> captures "Murzillka"
    # [WhiteElo "2186"] -> captures "2186"
    METADATA_PATTERNS: Dict[str, str] = {
        'Event': r'\[Event "([^"]+)"\]',
        'Site': r'\[Site "([^"]+)"\]',
        'Date': r'\[Date "([^"]+)"\]',
        'White': r'\[White "([^"]+)"\]',
        'Black': r'\[Black "([^"]+)"\]',
        'WhiteElo': r'\[WhiteElo "([^"]+)"\]',
        'BlackElo': r'\[BlackElo "([^"]+)"\]',
        'TimeControl': r'\[TimeControl "([^"]+)"\]',
        'Termination': r'\[Termination "([^"]+)"\]',
        'Result': r'\[Result "([^"]+)"\]',
    }

    # New separator: each game starts with a header block beginning with [Event "..."
    # We do NOT split the full file in memory; we stream line-by-line and start a new game at this header.
    GAME_START_TAG: str = '[Event "'

    # Pattern for PGN moves with eval/clock annotations
    # Captures: move number, chess notation, and optional annotations
    # Examples:
    # 1. e4 { [%eval 0.18] [%clk 0:10:00] }  # Full annotation
    # 1... c5 { [%clk 0:10:00] }              # Black move, clock only
    # 12. O-O { [%eval #3] }                  # Castling with mate eval
    #
    # Groups:
    # 1: Move number (e.g., "12")
    # 2: Chess move (e.g., "e4", "Nf3", "O-O", "e8=Q+, O-O-O+")
    # 3: Optional annotations (e.g., "[%eval 0.18] [%clk 0:10:00]")
    MOVE_PATTERN: str = (
        r'(\d+)(?:\.\.\.)?\.?\s*'
        r'('
            r'(?:'
                r'(?:[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8])'   # common move
                r'(?:=[QRBN])?'                           # promotion
                r'|(?:O-O(?:-O)?|0-0(?:-0)?)'             # castling
            r')'
            r'(?:\+\+?|\#)?'                              # check/mate
        r')'
        r'(?:[!?]{1,2})?'                                 # ignore ?, ??, ?!
        r'(?:\s*\{([^}]+)\})?'
    )

    # Pattern to extract eval from annotations
    # Example: { [%eval 0.18] [%clk 0:10:00] } -> captures "0.18"
    # Also handles mate notations: [%eval #3] -> captures "#3"
    EVAL_PATTERN: str = r'\[%eval\s+([^]]+)\]'

    # Pattern to extract clock from annotations
    # Example: { [%eval 0.18] [%clk 0:10:00] } -> captures "0:10:00"
    CLOCK_PATTERN: str = r'\[%clk\s+([^]]+)\]'

    # Pattern to remove game result from move section
    # Examples: "0-1", "1-0", "1/2-1/2"
    RESULT_PATTERN: str = r'\s*(?:0-1|1-0|1/2-1/2)\s*$'

    # Columns order for CSV output
    # Each row will contain: game metadata + move data
    CSV_COLUMNS: List[str] = [
        'GameId', 'Date', 'White', 'Black', 'WhiteElo', 'BlackElo',
        'TimeControl', 'Termination', 'GameResult',
        'HalfMove', 'Move', 'Eval', 'Clock'
    ]

    # ----------------------------
    # Construction / IO validation
    # ----------------------------

    def __init__(self, input_path: str, output_path: str) -> None:
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self._validate_paths()

        # Pre-compile regex patterns once to reduce repeated compilation overhead
        self._metadata_res: Dict[str, re.Pattern[str]] = {
            key: re.compile(pattern) for key, pattern in self.METADATA_PATTERNS.items()
        }
        self._move_re: re.Pattern[str] = re.compile(self.MOVE_PATTERN)
        self._eval_re: re.Pattern[str] = re.compile(self.EVAL_PATTERN)
        self._clock_re: re.Pattern[str] = re.compile(self.CLOCK_PATTERN)
        self._result_re: re.Pattern[str] = re.compile(self.RESULT_PATTERN)
        self._move_section_re1: re.Pattern[str] = re.compile(r'\n\n(.+?)\n\n\[', re.DOTALL)
        self._move_section_re2: re.Pattern[str] = re.compile(r'\n\n(.+)', re.DOTALL)

    def _validate_paths(self) -> None:
        """Validate input and output paths"""
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_path}")

        # We only support .pgn.zst (large-scale lichess dumps)
        if self.input_path.suffix != ".zst":
            raise ValueError(
                f"Invalid input file: {self.input_path}\n"
                f"Expected a .zst archive with PGN text inside (e.g., *.pgn.zst)"
            )

        # Create output directory if it doesn't exist
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # Streaming IO
    # ----------------------------

    @contextmanager
    def _open_zst_text_stream(self) -> Iterator[io.TextIOBase]:
        """
        Open .zst input as a UTF-8 text stream.

        Notes:
        - Uses streaming decompression (does not load the file into RAM).
        - errors='replace' makes the pipeline robust to occasional bad bytes.
        """
        f = open(self.input_path, "rb")
        try:
            dctx = zstd.ZstdDecompressor()
            stream_reader = dctx.stream_reader(f)
            text_stream = io.TextIOWrapper(
                stream_reader,
                encoding="utf-8",
                errors="replace",
                newline=""
            )
            try:
                yield text_stream
            finally:
                # Close wrapper first (flush/finish decoding), then underlying stream/file
                text_stream.close()
                stream_reader.close()
        finally:
            f.close()

    def _iter_game_texts(self, stream: io.TextIOBase) -> Iterator[Tuple[int, str]]:
        """
        Iterate over raw PGN game texts from a text stream.

        New data format:
        - games are separated by only one blank line,
          so we cannot rely on triple-newline separators.
        - each game reliably starts with [Event "..."] header.

        This method yields complete game blocks including headers + moves.
        """
        game_lines: List[str] = []
        game_idx = 0
        started = False

        for line in stream:
            # New game starts here
            if line.startswith(self.GAME_START_TAG):
                if started and game_lines:
                    game_idx += 1
                    yield game_idx, "".join(game_lines)
                    game_lines = []
                started = True

            if started:
                game_lines.append(line)

        # Flush the last game
        if started and game_lines:
            game_idx += 1
            yield game_idx, "".join(game_lines)

    # ----------------------------
    # Small parsing helpers
    # ----------------------------

    def _is_game_eligible(self, game_text: str) -> bool:
        """
        Fast pre-filter: keep only games whose [Event "..."] contains "blitz" (case-insensitive).
        This runs before any expensive move parsing.
        """
        m = self._metadata_res['Event'].search(game_text)
        if not m:
            return False
        return self.REQUIRED_EVENT_KEYWORD in m.group(1).lower()
    
    @staticmethod
    def _game_id_from_site(site: str) -> str:
        """
        Extract lichess game id from Site URL.
        Examples:
          "https://lichess.org/JRgTjrR2" -> "JRgTjrR2"
        """
        if not site:
            return ""
        site = site.strip().rstrip("/")
        last = site.split("/")[-1]
        return last.split("?")[0]

    @staticmethod
    def _parse_elo(elo_str: str) -> int:
        """Parse ELO rating string to integer"""
        if elo_str and elo_str.isdigit():
            return int(elo_str)
        return 0

    @staticmethod
    def parse_clock_to_seconds(clock_str: str) -> Optional[int]:
        """Convert clock string 'H:MM:SS' to total seconds"""
        if not clock_str:
            return None

        clock_str = clock_str.strip()
        parts = clock_str.split(':')

        # Example: "0:10:00" -> hours=0, minutes=10, seconds=0 -> returns 600
        if len(parts) == 3:
            hours, minutes, seconds = map(int, parts)
            return hours * 3600 + minutes * 60 + seconds

        raise ValueError(
            f"Invalid time format: '{clock_str}'. "
            f"Expected 'H:MM:SS', got {len(parts)} parts"
        )

    @staticmethod
    def parse_eval(eval_str: str) -> Optional[float]:
        """Parse evaluation string, handling checkmate notations"""
        if not eval_str:
            return None

        eval_str = eval_str.strip()

        # Handle checkmate notations
        # Examples: "#3" -> returns 100.0 (mate in 3 for white)
        #           "#-2" -> returns -100.0 (mate in 2 for black)
        if eval_str.startswith('#'):
            mate_in = int(eval_str[1:])
            return 100.0 if mate_in > 0 else -100.0

        try:
            return float(eval_str)
        except ValueError:
            return None

    # ----------------------------
    # Core extraction logic
    # ----------------------------

    def _extract_metadata(self, game_text: str) -> GameMetadata:
        """Extract metadata from game headers"""
        metadata: Dict[str, str] = {}

        for key, pattern in self.METADATA_PATTERNS.items():
            match = self._metadata_res[key].search(game_text)
            metadata[key] = match.group(1) if match else ''

        # New dumps do not provide GameId tag
        # We derive GameId from Site URL: [Site "https://lichess.org/<id>"]
        site = metadata.get('Site', '')
        game_id = self._game_id_from_site(site)

        white_elo = self._parse_elo(metadata.get('WhiteElo', ''))
        black_elo = self._parse_elo(metadata.get('BlackElo', ''))

        return GameMetadata(
            game_id=game_id,
            date=metadata.get('Date', ''),
            white=metadata.get('White', ''),
            black=metadata.get('Black', ''),
            white_elo=white_elo,
            black_elo=black_elo,
            time_control=metadata.get('TimeControl', ''),
            termination=metadata.get('Termination', ''),
            result=metadata.get('Result', '')
        )

    def _extract_moves(self, game_text: str) -> List[ChessMove]:
        """Extract moves with their evaluations and clocks from game text"""
        moves_data: List[ChessMove] = []

        # Find the move section (after headers)
        # Example: finds content between "\n\n" and next "\n\n[" or end of file
        move_section_match = self._move_section_re1.search(game_text)
        if not move_section_match:
            move_section_match = self._move_section_re2.search(game_text)

        if not move_section_match:
            return moves_data

        move_section = move_section_match.group(1)

        # Remove game result from the end
        # Example: removes trailing "0-1" from move section
        move_section = self._result_re.sub('', move_section)

        # Initialize half-move counter
        # Half-move = ply number (1 for first move, 2 for second, etc.)
        half_move_counter = 1

        for match in self._move_re.finditer(move_section):
            move = match.group(2).strip()
            annotations = match.group(3)

            eval_value: Optional[float] = None
            clock_value: Optional[int] = None

            if annotations:
                # Extract eval if present
                # Example: "[%eval 0.18]" -> eval_value = 0.18
                eval_match = self._eval_re.search(annotations)
                if eval_match:
                    eval_value = self.parse_eval(eval_match.group(1))

                # Extract clock if present
                # Example: "[%clk 0:10:00]" -> clock_value = 600
                clock_match = self._clock_re.search(annotations)
                if clock_match:
                    clock_value = self.parse_clock_to_seconds(clock_match.group(1))

            moves_data.append(ChessMove(
                move=move,
                eval=eval_value,
                clock=clock_value,
                half_move=half_move_counter
            ))

            half_move_counter += 1

        return moves_data

    def _iter_games(self, stream: io.TextIOBase) -> Iterator[Tuple[GameMetadata, List[ChessMove]]]:
        """Iterate over games in PGN stream with lightweight filtering and stats"""

        processed = 0
        skipped = 0
        seen = 0

        for game_idx, game_text in self._iter_game_texts(stream):
            seen += 1

            if not game_text.strip():
                skipped += 1
                continue

            if not game_text.lstrip().startswith(self.GAME_START_TAG):
                skipped += 1
                continue

            # Fast skip by cheap header-only check
            if not self._is_game_eligible(game_text):
                skipped += 1
                continue

            try:
                metadata = self._extract_metadata(game_text)

                if not metadata.game_id:
                    skipped += 1
                    continue

                moves = self._extract_moves(game_text)

                if not moves:
                    skipped += 1
                    continue

                processed += 1
                yield metadata, moves

            except Exception:
                skipped += 1
                continue

            # stats logging
            if seen % 10_000 == 0:
                print(
                    f"[STATS] Seen: {seen:,} | "
                    f"Processed: {processed:,} | "
                    f"Skipped: {skipped:,}"
                )

    # ----------------------------
    # Public API
    # ----------------------------

    def save_to_csv(self) -> None:
        """
        Convert .pgn.zst to csv.zst

        Important:
        - This method is fully streaming: it does NOT build a giant DataFrame in RAM.
        - It writes CSV rows incrementally and keeps only a small sample for preview.
        """
        print(f"Reading PGN archive: {self.input_path}")
        print(f"Writing CSV file: {self.output_path}")

        processed_games = 0
        total_moves = 0
        sample_rows: List[Dict] = []

        with self._open_zst_text_stream() as stream, open(self.output_path, "wb") as out_f_bin:
            cctx = zstd.ZstdCompressor(level=3)  # level 1..22
            with cctx.stream_writer(out_f_bin) as zst_writer:
                with io.TextIOWrapper(zst_writer, encoding="utf-8", newline="") as out_f_text:
                    writer = csv.DictWriter(out_f_text, fieldnames=self.CSV_COLUMNS)
                    writer.writeheader()

                    # Process each game: metadata + moves
                    processed_games = 0
                    total_moves = 0
                    sample_rows: List[Dict] = []

                    for metadata, moves in self._iter_games(stream):
                        for move_data in moves:
                            row = {
                                'GameId': metadata.game_id,
                                'Date': metadata.date,
                                'White': metadata.white,
                                'Black': metadata.black,
                                'WhiteElo': metadata.white_elo,
                                'BlackElo': metadata.black_elo,
                                'TimeControl': metadata.time_control,
                                'Termination': metadata.termination,
                                'GameResult': metadata.result,
                                'HalfMove': move_data.half_move,
                                'Move': move_data.move,
                                'Eval': move_data.eval,
                                'Clock': move_data.clock
                            }
                            writer.writerow(row)
                            total_moves += 1

                            if len(sample_rows) < 5:
                                sample_rows.append(row)

                        processed_games += 1
                        if processed_games % 100_000 == 0:
                            print(f"Processed {processed_games} games...")

        print(f"\nSuccessfully processed {processed_games} games")
        print(f"Total moves: {total_moves}")
        print(f"Saved CSV file: {self.output_path}")

        # Show sample
        if sample_rows:
            print("\nFirst 5 rows of the CSV:")
            df_sample = pd.DataFrame(sample_rows, columns=self.CSV_COLUMNS)
            print(df_sample.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert PGN chess games to CSV format')
    parser.add_argument('input', help='Input PGN file path (.pgn.zst)')
    parser.add_argument('output', help='Output CSV file path (.pgn.zst)')

    args = parser.parse_args()

    processor = PGNProcessor(args.input, args.output)
    processor.save_to_csv()
