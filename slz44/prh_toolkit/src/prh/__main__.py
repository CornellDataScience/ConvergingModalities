"""Command line: `prh validate results.json [...]`."""
import argparse
import json
import sys
from .schema import validate


def main(argv=None):
    parser = argparse.ArgumentParser(prog='prh', description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    check = sub.add_parser('validate', help='check results files against the shared schema')
    check.add_argument('files', nargs='+')
    args = parser.parse_args(argv)
    failed = False
    for name in args.files:
        try:
            with open(name, encoding='utf-8') as f:
                errors = validate(json.load(f))
        except (OSError, json.JSONDecodeError) as exc:
            errors = [str(exc)]
        print(f'{name}: ' + ('ok' if not errors else f'{len(errors)} problem(s)'))
        for e in errors:
            print('  -', e)
        failed = failed or bool(errors)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
