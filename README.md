# decision_pipeline

Trains interpretable classifiers on decisions extracted from an event log.

## Install and run

Python 3.9+.

```bash
pip install -e .
python -m decision_pipeline.core --input <log.csv> --config <config.json>
```

Flags:

- `--test <path>` use a separate test log instead of splitting `--input`.
- `--test-percentage <float>` test fraction when `--test` isn't given (default `0.2`).

## Two ways to find the decision point

Set `preprocess_config.type` in the config.

**`"log"`**: decision is located by activity names. List the activities that
mark the decision point and the activities that represent each outcome:

```json
"pre_decision_activities":   ["Decision Step"],
"post_decision_0_activities": ["Outcome A"],
"post_decision_1_activities": ["Outcome B"]
```

Example: `data/loan_log_config.json`.

**`"bpmn"`**: decision is located by a gateway in a BPMN model. Point at the
BPMN file and map its outgoing flows to outcomes:

```json
"bpmn_model_path":   "path/to/model.bpmn",
"target_gateway_id": "<gateway id>",
"outcome_mapping":   { "<flow id A>": 0, "<flow id B>": 1 }
```

Example: `data/loan_bpmn_config.json`.

## Config

Every option is documented in `decision_pipeline/config.schema.json`. See
`config.json` for a complete working example.

## Output

Everything is written to `evaluation_config.output_directory`: a comparison CSV,
classification reports, ROC plot, tree plots, and rule listings per model.
