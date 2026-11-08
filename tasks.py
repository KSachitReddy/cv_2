"""
tasks.py - the short commands of this project (registered in pyproject.toml under [project.scripts]).

    uv run bench     run the benchmark            (same options as benchmark.py, e.g. --n 200)
    uv run report    diagrams + Word report
    uv run slides    PowerPoint deck
    uv run build     diagrams + report + slides
    uv run demo      live demo in the browser     (uv run demo --image X.jpg = no-browser fallback)
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def bench():
    import benchmark
    benchmark.main()


def report():
    import build_report
    build_report.main()


def slides():
    import build_slides
    build_slides.main()


def build():
    report()      # also draws the diagrams
    slides()


def demo():
    if len(sys.argv) > 1:                     # e.g. --image photo.jpg -> command-line fallback
        import demo as demo_app
        demo_app.cli()
        return
    # Same as typing: streamlit run demo.py. Headless mode skips Streamlit's first-run "Email:" prompt;
    # we open the browser ourselves a moment later (set NO_BROWSER=1 to skip that).
    import threading
    import webbrowser
    from streamlit.web import cli as stcli
    port = 8501
    if not os.environ.get("NO_BROWSER"):
        threading.Timer(3.0, webbrowser.open, [f"http://localhost:{port}"]).start()
    sys.argv = ["streamlit", "run", os.path.join(HERE, "demo.py"), "--server.headless", "true",
                "--server.port", str(port), "--browser.gatherUsageStats", "false"]
    sys.exit(stcli.main())
