# First run

From the repository root, configure your provider credentials, then run:

```bash
python scripts/doctor.py --check-embedding --check-llm
bash examples/quickstart/run.sh "$PWD/output/quickstart"
```

The diagnostic loads the embedding model and sends a small structured request
to your selected LLM. The run script uses the existing `user_requirement.txt`
unchanged. It is an example of operating the CLI, not a reference solution.
Choose a new output directory for each run.

For your own prompt, copy `user_requirement.txt` to a separate file and pass it
with `python foambench_main.py --prompt_path /path/to/requirements.txt --output /path/to/new-output`.
See the [Docker quick start](../../README.md#quick-start-recommended-docker)
for mounting inputs and keeping results on the host.
