"""Entry point for triggering pipelines and serving the API.

    python main.py evaluate --limit 50    # score labelled images in raw_data
    python main.py predict --image x.jpg  # score a single image
    python main.py serve                  # start the FastAPI server
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from prayaas.config.configuration import settings
from prayaas.logger import logger


def cmd_evaluate(args: argparse.Namespace) -> None:
    from pipeline.evaluation_pipeline import EvaluationPipeline

    print(json.dumps(EvaluationPipeline().run(limit=args.limit), indent=2))


def cmd_predict(args: argparse.Namespace) -> None:
    from pipeline.prediction_pipeline import PredictionPipeline

    path = Path(args.image)
    if not path.exists():
        raise FileNotFoundError(f"No image at {path}")

    print(json.dumps(PredictionPipeline().predict_bytes(path.read_bytes()), indent=2))


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    uvicorn.run("app:app", host=settings.host, port=settings.port, reload=args.reload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="prayaas")
    sub = parser.add_subparsers(dest="command", required=True)

    evaluate = sub.add_parser("evaluate", help="Score the labelled images in artifacts/raw_data")
    evaluate.add_argument("--limit", type=int, help="Only score this many images (stratified)")
    evaluate.set_defaults(func=cmd_evaluate)

    predict = sub.add_parser("predict", help="Score a single image")
    predict.add_argument("--image", required=True, help="Path to a .jpg/.png file")
    predict.set_defaults(func=cmd_predict)

    serve = sub.add_parser("serve", help="Start the FastAPI server")
    serve.add_argument("--reload", action="store_true", help="Auto-reload on code changes")
    serve.set_defaults(func=cmd_serve)

    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    try:
        parsed.func(parsed)
    except Exception as e:
        logger.error("%s failed: %s", parsed.command, e)
        sys.exit(1)
