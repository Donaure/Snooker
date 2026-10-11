"""Build the Chinese documentation with strict reference checks."""

import argparse
from pathlib import Path
import subprocess
import sys


def main():
    docs = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=docs.parent / "outputs/docs/html")
    args = parser.parse_args()
    output = args.output.resolve()
    subprocess.run(
        [sys.executable, "-m", "sphinx", "-b", "html", "-n", "-W",
         "--keep-going", str(docs), str(output)],
        check=True,
    )


if __name__ == "__main__":
    main()
