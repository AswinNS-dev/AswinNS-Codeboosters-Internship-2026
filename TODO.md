# TODO - Fix LangChain/Transformers import crash

- [x] Investigated repo imports (`agent.py` uses `langchain_groq` -> `langchain_core` -> `transformers`).
- [x] Confirmed current environment has `transformers==5.10.2` with Python 3.13.
- [x] Pin compatible dependency versions in `requirements.txt` (transformers/tokenizers capped).
- [ ] Create a clean virtual environment (`.venv`) and reinstall pinned requirements.
- [ ] Verify by running `python app.py` (or `py app.py`) and confirm import no longer crashes.
- [ ] If it still crashes, iterate pins (especially `transformers`, `langchain-core`, `langchain-groq`).

