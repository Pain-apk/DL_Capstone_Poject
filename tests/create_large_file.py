with open("tests/large_test_file.jpg", "wb") as f:
    f.write(b"\xff\xd8\xff" + b"0" * (11 * 1024 * 1024))
print("Created tests/large_test_file.jpg (11 MB)")
