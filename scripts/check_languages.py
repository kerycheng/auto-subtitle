import argparse
import json
from autosub.language_validation import run_language_validation

parser = argparse.ArgumentParser()
parser.add_argument("manifest")
parser.add_argument("--model", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
args = parser.parse_args()
report = run_language_validation(args.manifest, args.model, args.output, args.device)
print(json.dumps({"status": report["status"], "by_language": report["by_language"]}))
