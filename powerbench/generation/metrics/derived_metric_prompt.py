
stage_1_prompt = r"""
You are an expert in power equipment monitoring and maintenance metrics. You are given the metric catalog (from metric_catalog.json) for a specific type of equipment, containing metric names, database field names, data types, units, discrete values, etc.

Your task: based on these "atomic metrics", propose a set of "derived / composite metrics" that have **well-established engineering or physical significance** — not arbitrary combinations.

Special note: "derived metrics" here are defined as "composite computation results of **multiple atomic metrics** at a **single point in time**" (e.g., ratios, differences, weighted combinations, etc.).
They must be multi-metric composites; single-metric statistics or trend computations belong to the time-series layer and are out of scope here.

## Input
Device type: {device_type}
Metric catalog (JSON array): {metric_catalog_json}

## Hard constraints (must be satisfied)
1. Output JSON only (no explanatory text of any kind).
2. Each derived metric must contain:
   - derived_name: Chinese or English name of the derived metric (concise and precise)
   - derived_key: suitable as a database field name / variable name (snake_case, ASCII alphanumeric and underscore)
   - description: physical meaning / engineering interpretation of the metric (1–3 sentences)
   - atomic_metrics: list of atomic metrics used (each item contains: db_field, indicator_name, unit)
   - formula: calculation formula (Python expression / pseudocode; specify which db_fields are used)
   - output_unit: output unit (empty string if dimensionless)
   - value_type: continuous / discrete / boolean
   - prerequisites: applicable preconditions or quality requirements (e.g., must be sampled at the same moment, denominator must not be zero, missing-value handling rules)
   - references: common engineering basis for this combination (e.g., "ratio-type oil DGA diagnosis", "load index / composite index", etc.; may be empty array)
3. Derived metrics must satisfy at least one of:
   - Industry-common ratio / index / composite indicator (e.g., oil DGA ratio types, load/composite indices, temperature rise / differential types, utilization-rate types, etc.)
   - Must be "single time-point, multi-metric composite"; time-window / trend / variance / range statistics are not allowed.
4. Do not generate metrics that require atomic metrics not present in the catalog.
5. If a metric has discrete values (non-empty discrete_values), do not treat it as a continuous quantity for arithmetic operations.
6. Prefer metrics whose units are consistent or can be reasonably converted; if conversion is needed, document it in formula and prerequisites.
7. Do not output duplicate or highly similar derived metrics (same meaning expressed differently counts as a duplicate).
8. The following types are prohibited (do not generate them at all):
   - Normalization / standardization / MinMax / arbitrarily weighted "composite indices" (unless there is a clear physical source listed in references)
   - Concepts unrelated to power equipment monitoring (e.g., "formaldehyde", etc.)
   - Metrics that require training a model / regression / black-box inference

## Output JSON Schema (must be strictly followed)
{{
  "device_type": "...",
  "derived_metrics": [
    {{
      "derived_name": "...",
      "derived_key": "...",
      "description": "...",
      "atomic_metrics": [
        {{"db_field": "...", "indicator_name": "...", "unit": "..."}}
      ],
      "formula": "...",
      "output_unit": "...",
      "value_type": "continuous",
      "prerequisites": ["..."],
      "references": ["..."]
    }}
  ]
}}
""".strip()


stage_2_prompt = r"""
You are a reviewer of power equipment monitoring and maintenance metrics. You will be given one derived metric (derived_metric) at a time and must judge whether it satisfies the rules.

## Input
Device type: {device_type}
Metric catalog (metric_catalog.json, array): {metric_catalog_json}
Derived metric under review (JSON): {derived_metric_json}

## Review criteria (must be satisfied)
1. Your output must be exactly one word: yes or no (no extra characters, explanation, or JSON).
2. Acceptance criteria (output "no" if any one fails):
   - Must be a composite computation of "a single time-point + multiple atomic metrics" (atomic_metrics must contain at least 2 entries)
   - Every db_field in atomic_metrics must exist in metric_catalog.json
   - formula may only reference db_fields present in atomic_metrics (no fields outside the catalog)
   - Discrete metrics must not participate in continuous arithmetic operations
   - output_unit / dimensionality must be reasonable; dimensionless may be an empty string
   - Normalization / standardization / score-type composite indices without a clear physical basis are not allowed
   - Concepts unrelated to power equipment are not allowed (e.g., "formaldehyde", etc.)
""".strip()
