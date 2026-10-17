import argparse
import json
from autosub.mixed_validation import run_mixed_validation

parser = argparse.ArgumentParser()
parser.add_argument("manifest")
parser.add_argument("--model", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
parser.add_argument("--strategy", choices=("fixed30-v1", "mixed-vad-v1"), default="mixed-vad-v1")
args = parser.parse_args()
result = run_mixed_validation(args.manifest, args.model, args.output, args.device, args.strategy)
print(json.dumps({"script_presence_passed": result["script_presence_passed"],
                  "regions": [(row["id"], row["expected_script_present"]) for row in result["regions"]]}))
