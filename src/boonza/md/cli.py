"""``boonza md``: the command line of the simulation workflow."""

from __future__ import annotations

import sys


def main(argv=None) -> int:
    from .config import parse_arguments, write_default_configuration
    from .prepare import describe_components
    from .run import run_workflow

    args = parse_arguments(argv)
    try:
        if args.write_default_config:
            write_default_configuration(args.write_default_config)
            print(f"Settings template written to {args.write_default_config}")
            return 0
        if args.list_components:
            print(describe_components(args))
            return 0
        run_workflow(args)
    except (ValueError, FileNotFoundError, NotADirectoryError, RuntimeError) as e:
        print(f"boonza md: error: {e}", file=sys.stderr)
        return 1
    return 0
