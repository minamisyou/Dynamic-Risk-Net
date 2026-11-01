"""Command-line interface for building and reviewing human-labelled DRN data."""
import argparse
import json

from .compat import load_config


def _command(subparsers, name, help_text):
    parser = subparsers.add_parser(name, help=help_text)
    # Accept --config both before and after the command without resetting a global value.
    parser.add_argument("--config", default=argparse.SUPPRESS)
    return parser


def build_parser():
    parser = argparse.ArgumentParser(prog="python -m drn")
    parser.add_argument("--config", default="config.yaml", help="path to config.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = _command(subparsers, "build", "build scene records from nuScenes")
    build.add_argument("--limit-scenes", type=int, help="process only the first N scenes")
    build.add_argument("--keep-all", action="store_true", help="also write records for rejected keyframes")
    build.add_argument("--only-tokens", default="", help="CSV file whose first column lists sample_token values")

    review = _command(subparsers, "review", "review retained keyframes")
    annotate = _command(subparsers, "annotate", "record independent annotator labels")
    annotate.add_argument("--prefill", default="", help="label set used as a starting draft")

    _command(subparsers, "consensus", "combine annotator labels and calculate agreement")
    _command(subparsers, "export", "write the human-label release files")
    _command(subparsers, "sheets", "write offline review sheets")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_config(args.config)

    if args.command == "build":
        from .build import read_token_list, run_build
        only = read_token_list(args.only_tokens) if args.only_tokens else None
        result = run_build(cfg, args.limit_scenes, args.keep_all, only_tokens=only)
        print(json.dumps(result, indent=2))
    elif args.command in ("review", "annotate"):
        from .annotate import create_app
        app = create_app(cfg, mode=args.command, prefill=getattr(args, "prefill", ""))
        app.run(host=cfg["annotation"]["host"], port=int(cfg["annotation"]["port"]), debug=False)
    elif args.command == "consensus":
        from .consensus import run_consensus, stats_markdown
        print(stats_markdown(run_consensus(cfg)))
    elif args.command == "export":
        from .export import run_export
        print(json.dumps(run_export(cfg), indent=2))
    elif args.command == "sheets":
        from .sheets import run_sheets
        print(json.dumps(run_sheets(cfg), indent=2))


if __name__ == "__main__":
    main()
