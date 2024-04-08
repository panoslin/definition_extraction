"""
Helper functions.
"""

import json
import os


def save_config(
        config,
        path,
        verbose=True
):
    with open(path, 'w') as outfile:
        json.dump(config, outfile, indent=2)
    if verbose:
        print("Config saved to file {}".format(path))
    return config


def print_config(
        config
):
    info = "Running with the following configs:\n"
    for k, v in config.items():
        info += f"\t{k} : {v}\n"
    print("\n" + info + "\n")
    return


class FileLogger:
    """
    A file logger that opens the file periodically and write to it.
    """

    def __init__(
            self,
            filename,
            header=None
    ):
        self.filename = filename
        if os.path.exists(filename):
            # remove the old file
            os.remove(filename)

        if header is not None:
            with open(filename, 'w') as out:
                print(header, file=out)

    def log(
            self,
            message
    ):
        with open(self.filename, 'a') as out:
            print(message, file=out)
