"""
Project entry point for the DQN Tool Selection Agent skeleton.

This script verifies that the project package can import the action space.
"""

from src.constants import ACTIONS


def main():
    """Print a short initialization message and the configured action space."""
    print("DQN Tool Selection Agent project initialized.")
    print(f"Action space: {ACTIONS}")


if __name__ == "__main__":
    main()
