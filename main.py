"""Transkrypcja wielu plików FLAC do jednego pliku VTT.

Wymaga: openai-whisper (oficjalny pakiet).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import whisper
import tempfile
from pathlib import Path
from typing import Iterable

try:
	from pydub import AudioSegment, silence
except Exception as exc:  # pragma: no cover - helpful runtime message
	raise RuntimeError(
		"pydub is required for VAD trimming. Install with: pip install pydub "
		"and ensure ffmpeg is available on PATH"
	) from exc


def format_timestamp(seconds: float) -> str:
	"""Format czasu do WEBVTT: HH:MM:SS.mmm."""
	milliseconds = int(round(seconds * 1000))
	hours = milliseconds // 3_600_000
	minutes = (milliseconds % 3_600_000) // 60_000
	secs = (milliseconds % 60_000) // 1_000
	ms = milliseconds % 1_000
	return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


def speaker_from_filename(path: Path) -> str:
	"""Wyznacz nazwę rozmówcy na podstawie nazwy pliku."""
	name = path.stem.replace("_", " ").strip()
	if not name:
		return "Rozmówca"
	return name[:1].upper() + name[1:]


def iter_flac_files(folder: Path) -> Iterable[Path]:
	"""Zwraca posortowaną listę plików .flac z folderu."""
	return sorted(folder.glob("*.flac"))


def write_segments_to_vtt(vtt_path: Path, speaker: str, segments: list[dict]) -> None:
	"""Zapisuje segmenty do pliku VTT z prefixem rozmówcy."""
	with vtt_path.open("w", encoding="utf-8") as vtt:
		vtt.write("WEBVTT\n\n")
		for segment in segments:
			start = format_timestamp(segment["start"])
			end = format_timestamp(segment["end"])
			text = segment["text"].strip()

			vtt.write(f"{start} --> {end}\n")
			vtt.write(f"[{speaker}] {text}\n\n")


def write_merged_vtt(vtt_path: Path, merged_segments: list[dict]) -> None:
	"""Zapisuje scalony plik VTT z posortowanymi wpisami."""
	with vtt_path.open("w", encoding="utf-8") as vtt:
		vtt.write("WEBVTT\n\n")
		for item in merged_segments:
			start = format_timestamp(item["start"])
			end = format_timestamp(item["end"])
			text = item["text"].strip()
			speaker = item["speaker"]

			vtt.write(f"{start} --> {end}\n")
			vtt.write(f"[{speaker}] {text}\n\n")


def transcribe_files_to_vtt(audio_dir: Path, output_vtt: Path) -> None:
	"""Transkrybuje wszystkie pliki FLAC i zapisuje jeden plik VTT.

	Dodatkowo zapisuje osobny plik VTT dla każdego pliku audio.
	"""
	files = list(iter_flac_files(audio_dir))
	if not files:
		print(f"Brak plików .flac w folderze: {audio_dir}")
		return

	model = whisper.load_model("turbo", device="cuda")
	print("Ładowanie modelu Whisper: turbo (CUDA, fp16)")

	output_vtt.parent.mkdir(parents=True, exist_ok=True)
	merged_segments: list[dict] = []

	total_files = len(files)
	for index, audio_file in enumerate(files, start=1):
		speaker = speaker_from_filename(audio_file)
		print(f"[{index}/{total_files}] Przetwarzanie: {audio_file.name} -> [{speaker}]")

		# Apply VAD trimming to remove long silences before transcription
		def vad_trim_audio(path: Path, min_silence_len: int = 500, silence_thresh: int = -40, padding: int = 200) -> Path:
			"""Trim long silent parts from `path` and return path to trimmed temp file.
			Keeps a small `padding` (ms) around nonsilent regions to avoid cutting words.
			Requires pydub and ffmpeg available in PATH.
			"""
			audio = AudioSegment.from_file(str(path))
			nonsilent_ranges = silence.detect_nonsilent(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)

			if not nonsilent_ranges:
				# No voice detected — return original file (skip trimming)
				return path

			# Build combined audio from nonsilent ranges with small padding
			parts = []
			for start_ms, end_ms in nonsilent_ranges:
				start = max(0, start_ms - padding)
				end = min(len(audio), end_ms + padding)
				parts.append(audio[start:end])

			trimmed = parts[0]
			for part in parts[1:]:
				# insert a short gap between concatenated parts to avoid words running together
				trimmed += AudioSegment.silent(duration=padding) + part

			# Export to temporary file (FLAC)
			tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".flac")
			tmp_name = tmp.name
			tmp.close()
			trimmed.export(tmp_name, format="flac")
			return Path(tmp_name)

		processed_file = vad_trim_audio(audio_file)
		result = model.transcribe(
			str(processed_file),
			language="Polish",
			condition_on_previous_text=False,
			fp16=True,
			verbose=False,
			# no_speech_threshold =1.0,
			temperature=0.0,
			no_speech_threshold=0.7,
			logprob_threshold=-0.8,
			compression_ratio_threshold=2.0,
			
			# beam_size=3,
            # best_of=3,
            # --language Polish \
            # --device cuda \
            # --fp16 True \
            # --temperature 0 \
            # --condition_on_previous_text False \
            # --beam_size 3 \
            # --best_of 3
		)

		segments = result.get("segments", [])
		print(f"   Segmenty: {len(segments)}")

		per_file_vtt = audio_dir / f"{audio_file.stem}.vtt"
		write_segments_to_vtt(per_file_vtt, speaker, segments)
		print(f"   Zapisano osobny plik: {per_file_vtt.name}")

		for seg_index, segment in enumerate(segments, start=1):
			merged_segments.append(
				{
					"start": segment["start"],
					"end": segment["end"],
					"text": segment["text"],
					"speaker": speaker,
				}
			)

			if seg_index % 50 == 0 or seg_index == len(segments):
				print(f"   Postęp segmentów: {seg_index}/{len(segments)}")

		# Clean up temporary processed file if created
		try:
			if processed_file != audio_file and processed_file.exists():
				processed_file.unlink()
		except Exception:
			pass

	merged_segments.sort(key=lambda item: (item["start"], item["end"]))
	write_merged_vtt(output_vtt, merged_segments)

	print(f"Gotowe. Zapisano: {output_vtt}")


def main() -> None:
	base_dir = Path(__file__).resolve().parent
	audio_dir = base_dir / "audio"
	output_vtt = base_dir / "output.vtt"

	transcribe_files_to_vtt(audio_dir, output_vtt)


if __name__ == "__main__":
	main()
