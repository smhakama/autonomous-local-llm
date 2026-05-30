import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hello import greet


def test_greets_alice():
    assert greet("Alice") == "Hello, Alice!"


def test_greets_bob():
    assert greet("Bob") == "Hello, Bob!"


def test_greets_empty():
    assert greet("") == "Hello, !"
