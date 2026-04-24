---
name: add-method
description: Procedure for adding a new analysis method or data source to the llm_trader pipeline.
argument-hint: "[method name and brief description]"
---

Follow these steps in order to add the new method ($ARGUMENTS):

1. **Update the code** — implement the new or updated method in the codebase.

2. **Add to the email** — if the method is not already represented in the email report, add it to `src/notifications/email_sender.py`.

3. **Update `CLAUDE.md` and `README.md`** — update the md files with the new code changes and documentation.

4. **Run `main.py`** and enable the minimal amount of data fetching to be abl to test — execute the pipeline and check the output.

5. **Debug if needed** — if there are errors, diagnose and fix them, then re-run until clean.

6. **Put back the data fetching value to where it was before the test** — restore the data fetching settings to their original values.
