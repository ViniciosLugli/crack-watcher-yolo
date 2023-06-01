import sys

import train
import run


class Commands:
    @staticmethod
    def help():
        print("Available commands:")
        for command in COMMANDS:
            print(f"  {command} - {COMMANDS[command]['about']}")
        print("")
        print("Usage:")
        print("  python main.py <command>")
        print("")
        print("Example:")
        print("  python main.py train")

    @staticmethod
    def run(image_path):
        print("Running model...")
        run.run(image_path)

    @staticmethod
    def train():
        print("Training model...")
        train.train()
        print("Training complete!")


COMMANDS = {
    "help": {
        "about": "Prints help message of project.",
        "function": Commands.help
    },
    "run": {
        "about": "Runs the model.",
        "function": Commands.run
    },
    "train": {
        "about": "Trains the model.",
        "function": Commands.train
    }
}


def main():
    command = sys.argv[1]
    args = sys.argv[2:]
    if command in COMMANDS:
        COMMANDS[command]["function"](args)
    else:
        print(f"Command '{command}' not found.")
        Commands.help()


if __name__ == "__main__":
    main()
