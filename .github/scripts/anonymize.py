"""Script to anonymize research paper given authors/emails"""
import argparse
import pathlib
from fileinput import FileInput

import tomllib


def find_replace(filepaths: list[str], find="", replace=""):
    if not find or not replace:
        print("No find or replace targets")
        return

    for line in FileInput(filepaths, inplace=True, encoding="utf-8"):
        print(line.replace(find, replace), end="")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Anonymize", description="Anonymize files")
    parser.add_argument("-f", "--files", nargs="+", help="Input files", required=True)
    files = parser.parse_args().files
    filepaths = [file for file in files if pathlib.Path(file).is_file()]

    with open("pyproject.toml", "rb") as f:
        data = tomllib.load(f)

        for i, author_dict in enumerate(data["project"]["authors"], start=1):
            find_replace(filepaths, author_dict["name"], f"Anonymous Author {i}")
            find_replace(filepaths, author_dict["email"], f"opendataval+{i}@gmail.com")
