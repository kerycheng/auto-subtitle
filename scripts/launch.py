if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2 and sys.argv[1] == "--self-test":
        from autosub.diagnostics import self_test
        self_test(sys.argv[2], *(sys.argv[3:6]))
    elif len(sys.argv) > 1 and sys.argv[1] == "--language-check":
        if len(sys.argv) not in (5, 6):
            raise SystemExit("--language-check requires manifest, local model directory, output directory, optional cpu/cuda")
        from autosub.language_validation import run_language_validation
        run_language_validation(sys.argv[2], sys.argv[3], sys.argv[4], *(sys.argv[5:6]))
    elif len(sys.argv) > 1 and sys.argv[1] == "--mixed-check":
        if len(sys.argv) not in (5, 6, 7):
            raise SystemExit("--mixed-check requires manifest, local model, output, optional cpu/cuda, optional strategy")
        from autosub.mixed_validation import run_mixed_validation
        run_mixed_validation(sys.argv[2], sys.argv[3], sys.argv[4], *(sys.argv[5:7]))
    else:
        from autosub.app import main
        main()
