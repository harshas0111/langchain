import sys
print("--- ACTIVE PYTHON INTERPRETER ---")
print(sys.executable)
print("\n--- LOOKING FOR PACKAGES IN ---")
for path in sys.path:
    print(path)
