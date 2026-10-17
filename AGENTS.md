# Repository work rules

- Commit application code, reusable tools, synthetic unit tests, dependency files and public usage documentation only.
- Keep private media, transcripts, subtitles, user projects, local paths, study plans, stage reports and experiment outputs out of Git, including commit messages.
- Never force-add ignored local data. Local test authorization does not authorize publication.
- Keep local research under ignored directories or outside the repository. Do not add its results or notes to README or commit bodies.
- Write commit messages about the concrete code change; omit phase numbers, material identities and research outcomes.
- Before committing, run appropriate tests and `python scripts/check_repository.py` on the staged index, then inspect the staged diff.
- Do not rewrite public remote history without explicit authorization. Old local private archives must stay outside the shareable repository.
