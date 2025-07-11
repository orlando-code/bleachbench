def test_import_bleachbench():
    try:
        import bleachbench
    except ImportError as e:
        assert False, f"Import failed: {e}"
