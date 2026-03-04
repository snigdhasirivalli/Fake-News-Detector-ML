"""
verify_setup.py
---------------
Run this script to confirm all required libraries are installed
and accessible. It will print the version of each library or a
clear error message if something is missing.

Usage:
    python verify_setup.py
"""

import sys

REQUIRED = [
    "pandas",
    "sklearn",       # scikit-learn is imported as sklearn
    "streamlit",
    "nltk",
    "joblib",
    "numpy",
    "matplotlib",
    "seaborn",
]

def check_library(name: str) -> tuple[bool, str]:
    """Try to import a library and return its version."""
    try:
        import importlib
        mod = importlib.import_module(name)
        version = getattr(mod, "__version__", "version unknown")
        return True, version
    except ImportError:
        return False, "NOT INSTALLED"


def main():
    print("=" * 55)
    print("  🔍  Fake News Detection — Environment Check")
    print("=" * 55)
    print(f"  Python version : {sys.version.split()[0]}")
    print("-" * 55)

    all_ok = True
    for lib in REQUIRED:
        ok, version = check_library(lib)
        status = "✅" if ok else "❌"
        display_name = "scikit-learn" if lib == "sklearn" else lib
        print(f"  {status}  {display_name:<18}  {version}")
        if not ok:
            all_ok = False

    print("-" * 55)
    if all_ok:
        print("  ✅  All libraries are installed. You're good to go!")
    else:
        print("  ❌  Some libraries are missing.")
        print("      Run:  pip install -r requirements.txt")
    print("=" * 55)


if __name__ == "__main__":
    main()
