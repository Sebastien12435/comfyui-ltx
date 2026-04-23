import argparse
from pathlib import Path

from transformers import AutoModelForCausalLM


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a Transformers CausalLM model to safetensors."
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Local path or Hugging Face model ID (for example: google/gemma-3-12b-it).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Directory where converted model files will be saved.",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.source,
        torch_dtype="auto",
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    model.save_pretrained(str(output_path), safe_serialization=True)

    print(f"Saved safetensors model to: {output_path}")


if __name__ == "__main__":
    main()
