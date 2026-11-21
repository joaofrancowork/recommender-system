"""recsys download | evaluate | train | serve"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from recsys.data import DATA_DIR, download
from recsys.evaluation import to_markdown
from recsys.pipeline import build_models, evaluate_all, prepare, save_artifacts, tune


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="recsys")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("download")
    for name in ("evaluate", "train"):
        p = sub.add_parser(name)
        p.add_argument("--data", type=Path, default=DATA_DIR)
        p.add_argument("--out", type=Path, default=Path("artifacts"))
        p.add_argument("--tune", action="store_true", help="Grid search on a validation split")
        p.add_argument(
            "--content",
            choices=["embeddings", "genres"],
            default="embeddings",
            help="Item representation for the content and hybrid models",
        )
    serve = sub.add_parser("serve")
    serve.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.command == "download":
        print(download())
        return 0
    if args.command == "serve":  # pragma: no cover
        import uvicorn

        from recsys.api import create_app

        uvicorn.run(create_app(), host="0.0.0.0", port=args.port)
        return 0

    ds = prepare(download(args.data))
    params = tune(ds) if args.tune else {}
    embeddings = None
    if args.content == "embeddings":  # pragma: no cover - needs the fastembed model download
        from recsys.embeddings import embed_movies

        embeddings = embed_movies(ds.movies, args.out / "item_embeddings.npy")
    models = build_models(ds, embeddings, params)
    reports = evaluate_all(ds, models)
    table = to_markdown(reports)
    print(json.dumps(params) if params else "default hyper-parameters")
    print(table)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "evaluation.md").write_text(table + "\n")
    if args.command == "train":
        save_artifacts(ds, models, args.out, params)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
