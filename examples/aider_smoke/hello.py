def greet(name):
    # BUG: name is ignored
    return f"Hello, {name}!"


if __name__ == "__main__":
    print(greet("Makoto"))
