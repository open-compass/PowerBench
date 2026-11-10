stage_1_prompt = """
You are an expert in power equipment maintenance knowledge modeling, skilled at deriving the common maintenance metrics, special maintenance metrics, and special concerns of equipment from the device list and existing metric clues.

[Task Objective]
The user will provide two kinds of input:
1. Device table
2. Metric summary json (containing part of the existing fields, existing metrics, value ranges, applicable conditions, or concern clues)

Please note:
- The device-table fields in the input are mainly used to describe device instances and configuration differences.
- The json in the input also mainly provides auxiliary clues, used to decide whether certain devices need extra maintenance metrics, or whether particular existing metrics or risk directions deserve special attention.
- These input fields, existing metrics and value ranges are not equivalent to the final complete maintenance metric system.
- Your task is to derive, on top of these inputs, the maintenance metric system that this type of equipment truly needs to monitor.
- This stage does no quantification, generates no anomaly judgment rules, no formulas, no thresholds.
- However, this stage must clearly describe the "data type (numeric/discrete) and value constraints (range/enumeration)" of each metric, so that the next stage can generate executable rules.

[Your Task]
1. Identify which fields in the device table are "instance-locating fields".
2. Identify which fields are "auxiliary classification fields".
3. Derive the "common maintenance metrics" generally applicable to this type of equipment.
4. Based on the auxiliary classification fields, decide whether "special maintenance metrics" need to be added.
5. Based on the auxiliary classification fields, decide whether there are existing metrics, states, or risk points that deserve special attention, and output "special concerns".
6. Output a structured maintenance metric system for the next stage to generate anomaly judgment rules.

[Concept Definitions]
1. Instance-locating fields:
   Used only to identify device instances, e.g., device name, device ID, location, etc.; not used to trigger new maintenance metrics.

2. Auxiliary classification fields:
   Used to distinguish device configuration differences, e.g., power supply type, cooling method, monitoring method, voltage regulation method, bushing type, etc.
   These fields are not maintenance metrics themselves; they only help decide whether certain devices need special maintenance metrics or special concerns.

3. Common maintenance metrics:
   Maintenance metrics generally applicable to and jointly monitored for this type of equipment.

4. Special maintenance metrics:
   Maintenance metrics that need to be added only when certain auxiliary classification fields take certain values.

5. Special concerns:
   Not necessarily a new maintenance metric, but existing metrics, a certain state, or a certain risk direction that deserves special attention.
   This stage describes them in text only, without quantification.

[Input]
Device type:
{device_type}

Device table (markdown):
{device_table_markdown}

Metric summary JSON:
{indicator_summary_json}

[Output Format]
Output strict JSON:

{
  "device_type": "",
  "instance_fields": [],
  "auxiliary_classification_fields": [],
  "common_maintenance_metrics": [
    {
      "metric_name": "",
      "metric_definition": "",
      "data_type": "numeric|discrete|text",
      "unit": "",
      "value_range": "",
      "discrete_values": [],
      "remarks": "If discrete, list common values in discrete_values; if numeric, give an engineering range expression in value_range."
    }
  ],
  "special_maintenance_metrics": [
    {
      "metric_name": "",
      "metric_definition": "",
      "data_type": "numeric|discrete|text",
      "unit": "",
      "value_range": "",
      "discrete_values": [],
      "trigger_conditions": [
        {
          "field_name": "",
          "field_values": []
        }
      ]
    }
  ],
  "special_concerns": [
    {
      "concern_description": "",
      "time_series_note": "May be empty; if it truly relates to duration/trend/fluctuation, a textual hint is allowed. The concrete time window and computation are left to the second stage.",
      "trigger_conditions": [
        {
          "field_name": "",
          "field_values": []
        }
      ]
    }
  ]
}

[Output Requirements]
1. "instance_fields" should output only fields genuinely used to identify device instances.
2. "auxiliary_classification_fields" should output only fields genuinely used to distinguish configuration differences and that can affect special maintenance metrics or special concerns.
3. "common_maintenance_metrics" should cover the main monitoring objects generally applicable to this type of equipment.
4. "special_maintenance_metrics" must be metrics added on top of the common maintenance metrics, not duplicates of existing metrics.
5. The trigger_conditions of "special_maintenance_metrics" must explicitly state which field and which values they correspond to.
6. "special_concerns" only needs a textual description; do not write formulas, thresholds, or judgment rules.
7. The trigger_conditions of "special_concerns" must likewise explicitly state which field and which values they correspond to.
8. Metric names should preferentially inherit existing names from the input json; if new maintenance metrics need to be added, names should be as standard, concise and engineering-like as possible.
9. **Atomic metrics (hard constraint)**: Every maintenance metric must correspond to **one** measurable point or state variable that can be stored/monitored independently. Do not bundle multiple measurement points into a single "metric_name" (e.g., "Oil DGA (total hydrocarbons, hydrogen, acetylene, etc.)" or "body and accessory pressure/density"); if there really are multiple items, they must be split into separate metrics. The name may annotate spec/alias in parentheses, but the parentheses must not enumerate multiple independent measurement points with separators.
10. At this stage do not output any quantitative expression, formula, rule expression, threshold parameter, or anomaly conclusion.
11. Prefer outputting "numeric" metrics whenever possible (they support arithmetic/rate-of-change computation); output "discrete" only when the quantity is objectively a state.
12. For discrete metrics, common values must be listed in "discrete_values" (e.g., normal/abnormal/offline/interrupted/alarm, etc.).
13. For numeric metrics, a reasonable expression must be given in "value_range" (e.g., 0~100%, ±10% of rated value, per manufacturer parameters, per standards, etc.), and "unit" must be filled in (may be left empty if unknown).
14. This stage does not need to output sampling frequency or time window; such time-series computation information is given in the second-stage rules.
15. Output JSON only, no extra explanation.

[Additional Constraints]
1. Do not output auxiliary classification fields themselves as maintenance metrics.
2. Do not use instance-locating fields as the basis for triggering special maintenance metrics.
3. Do not output an "inclusion reason".
4. Do not disguise "special concerns" as new metrics; if it is essentially just focused attention on existing metrics, existing states, or existing risk directions, it belongs in "special_concerns".
5. If the existing metrics in the input are only clues but insufficient to cover this type of equipment's common monitoring needs, you may add reasonable common or special maintenance metrics.
6. If the trigger of a special maintenance metric or special concern depends on multiple fields, multiple fields may be listed in "trigger_conditions".
7. Do not output composite names that bundle two physical quantities into one, such as "pressure/density" or "temperature/humidity"; split them into independent metrics (e.g., "body pressure", "bushing pressure"), or make it an explicitly single state variable name.
8. Output JSON only, no markdown, no explanatory text.

[Example Notes]
The following example is only for illustrating the input/output structure, field roles, and mapping relationships.
The device names, field values, metric names, and concern content in the example should not be mechanically copied to other device types.

[Example Input]
Device type:
IPS-3000

Device table (markdown):
| Device Name | Device ID | Power Type | Monitoring Method | Communication Protocol | Capacity Class |
|---|---|---|---|---|---|
| IPS-3000-001 | IPS-3000-001 | AC/DC hybrid | Smart monitoring | IEC 61850 | Large |
| IPS-3000-002 | IPS-3000-002 | DC power | Remote monitoring | Modbus | Medium |
| IPS-3000-003 | IPS-3000-003 | AC power | Local monitoring | DNP3 | Small |

Metric summary JSON:
{
  "metric_list": [
    {
      "metric_name": "Input voltage",
      "metric_description": "Voltage on the device input side",
      "reference_range": "±10% of rated value",
      "applicable_condition": "All devices"
    },
    {
      "metric_name": "Output voltage",
      "metric_description": "Voltage on the device output side",
      "reference_range": "±10% of rated value",
      "applicable_condition": "All devices"
    },
    {
      "metric_name": "Communication status",
      "metric_description": "Health status of the communication link",
      "reference_range": "normal/abnormal",
      "applicable_condition": "All devices"
    },
    {
      "metric_name": "Battery voltage",
      "metric_description": "Terminal voltage of the battery pack",
      "reference_range": "Set per model",
      "applicable_condition": "Devices with DC power"
    },
    {
      "metric_name": "Remote telemetry status",
      "metric_description": "Status of the remote monitoring link",
      "reference_range": "normal/abnormal",
      "applicable_condition": "Remote or smart monitoring devices"
    }
  ]
}

[Example Output]
{
  "device_type": "IPS-3000",
  "instance_fields": ["Device Name", "Device ID"],
  "auxiliary_classification_fields": ["Power Type", "Monitoring Method", "Communication Protocol", "Capacity Class"],
  "common_maintenance_metrics": [
    {
      "metric_name": "Input voltage",
      "metric_definition": "Supply voltage on the device input side, reflecting whether the upstream power supply state is normal."
    },
    {
      "metric_name": "Output voltage",
      "metric_definition": "Supply voltage on the device output side, reflecting the device output power supply state."
    },
    {
      "metric_name": "Communication status",
      "metric_definition": "Status of the communication link between the device and the host system or associated modules."
    }
  ],
  "special_maintenance_metrics": [
    {
      "metric_name": "Battery voltage",
      "metric_definition": "Voltage at the battery pack output terminal, reflecting the backup power supply health.",
      "trigger_conditions": [
        {
          "field_name": "Power Type",
          "field_values": ["AC/DC hybrid", "DC power"]
        }
      ]
    },
    {
      "metric_name": "Remote telemetry status",
      "metric_definition": "Status of the telemetry link between the device and the remote monitoring system.",
      "trigger_conditions": [
        {
          "field_name": "Monitoring Method",
          "field_values": ["Remote monitoring", "Smart monitoring"]
        }
      ]
    }
  ],
  "special_concerns": [
    {
      "concern_description": "For devices configured with remote or smart monitoring, special attention should be paid to unstable communication links, telemetry interruptions, and abnormal remote state reporting.",
      "trigger_conditions": [
        {
          "field_name": "Monitoring Method",
          "field_values": ["Remote monitoring", "Smart monitoring"]
        }
      ]
    },
    {
      "concern_description": "For devices configured with DC power, special attention should be paid to declining backup power capability, battery depletion, and supply stability during power switching.",
      "trigger_conditions": [
        {
          "field_name": "Power Type",
          "field_values": ["AC/DC hybrid", "DC power"]
        }
      ]
    }
  ]
}
"""


stage_2_prompt = """
You are an expert in power maintenance fact-checking rule modeling, skilled at turning maintenance metrics and concerns into executable, explainable, quantifiable anomaly judgment rules, used to judge whether the data and conclusions in maintenance and fault reports are accurate.

[Task Objective]
The user will provide the maintenance metric system output by stage one, including:
1. Common maintenance metrics
2. Special maintenance metrics
3. Special concerns

[Your Task]
Based on this content:
1. Generate anomaly judgment rules for each common maintenance metric.
2. Generate anomaly judgment rules for each special maintenance metric.
3. Generate a judgment approach for each special concern.
4. For quantifiable objects, preferentially give formulas, inequalities, boolean expressions, or logic-combination expressions.
5. For cases where a threshold cannot be uniformly determined, clearly state the parameter source (set per rated value, model, standard, or manufacturer parameters).
6. Output a structured rule result usable for fact-checking.
7. You must clearly write the time-series-related information: the time window used under the default hourly sampling, the statistics (mean/max/slope/rate-of-change, etc.), and the influence of the time span on the judgment.

[Quantitative Expression Requirements]
For each rule, output according to the following priority:

1. If it can be directly expressed as a mathematical formula or inequality, output the formula or inequality.
2. If it is a state judgment, output a boolean expression or set-membership judgment expression.
3. If it is a joint judgment over multiple metrics, output a logic-combination expression connected by AND / OR / NOT.

[Input]
Device type:
{device_type}

Stage-one output JSON:
{stage1_output_json}

[Output Format]
Output strict JSON:

{
  "device_type": "",
  "anomaly_rules": [
    {
      "rule_id": "K1",
      "object_name": "",
      "object_type": "common_maintenance_metric|special_maintenance_metric|special_concern",
      "applicable_conditions": [
        {
          "field_name": "",
          "field_values": []
        }
      ],
      "pointwise_judgment": {
        "single_point_trigger": true,
        "consecutive_points": 1,
        "note": "Default is hourly sampling; if consecutive_points > 1, the rule expression must explicitly reflect continuity (e.g. consec_3h(cond)=true)"
      },
      "time_series_requirements": {
        "requires_long_observation": true,
        "window_length": "1h|6h|24h|7d",
        "statistics": ["mean|max|min|std|rate_of_change|slope|occurrence_count|duration"]
      },
      "input_metrics": [],
      "derived_metric_definitions": [
        {
          "derived_metric_name": "",
          "formula": "",
          "note": ""
        }
      ],
      "judgment_description": "",
      "rule_expression": "",
      "threshold_parameters": [
        {
          "parameter_name": "",
          "meaning": "",
          "parameter_value": "To be checked in manufacturer manual/nameplate/standard"
        }
      ]
    }
  ]
}

[Field Descriptions]
1. object_name:
   Corresponds to one common maintenance metric, special maintenance metric, or special concern from stage one.

2. object_type:
   Can only be chosen from the following three:
   - common_maintenance_metric
   - special_maintenance_metric
   - special_concern

3. applicable_conditions:
   The applicable scope must be expressed in the structured form "field_name + field_values":
   - If applicable to all devices: applicable_conditions should be [] (empty array)
   - If it depends on a configuration/classification field: list one or more fields and their value ranges in applicable_conditions
   For special maintenance metrics or special concerns from stage one, their trigger-condition meaning should be inherited and rewritten as field conditions.

4. input_metrics:
   The metric names that this rule truly depends on for judgment;
   for a special concern, multiple metrics may be input.

5. derived_metric_definitions:
   If an intermediate quantity needs to be constructed first, such as deviation rate, temperature rise, fluctuation rate, rate of change, etc., define the derived metric and its formula first;
   if not needed, it may be an empty array.

6. judgment_description:
   A concise, clear textual description of the rule, stating "under what conditions it is judged abnormal".

7. rule_expression:
   Prefer filling in an executable expression, e.g., formula, inequality, boolean expression, logic-combination expression;
   if it truly cannot be quantified, it may be left empty, but this must be made clear in judgment_description;
   in principle an expression should be given whenever possible.

8. threshold_parameters:
   If the expression involves rated values, thresholds, time windows, manufacturer parameters, etc., list them here.

9. pointwise_judgment:
   Even for a pointwise rule, it must be clear whether it is "single-point trigger" or "consecutive-N-point trigger" (N counted in hourly sampling points).
   If consecutive_points > 1, the rule expression must reflect the continuity requirement (e.g. consec_3h(cond)=true).

10. time_series_requirements:
    Needed only when the rule is not a "pointwise judgment" (e.g., it needs trend, persistence, frequency, sliding-window statistics, etc.);
    if it is a pointwise rule (single-point threshold / single occurrence state), this field may be omitted or set to an empty object {}.

[Output Requirements]
1. Must cover all "common maintenance metrics" from stage one.
2. Must cover all "special maintenance metrics" from stage one.
3. Must cover all "special concerns" from stage one.
4. For quantifiable objects, must preferentially give formulas, inequalities, boolean expressions, or logic-combination expressions.
5. A "special concern" may be a single metric or a combination of multiple metrics, a kind of abnormal trend, or a kind of key risk.
6. The metric names in input_metrics must be consistent with the metric names from stage one.
7. Do not directly use the auxiliary classification fields from stage one as rule judgment objects, unless a field has itself been explicitly classified as a maintenance metric in stage one.
8. For numeric metrics: the rule expression should preferentially use time-window statistics (e.g. avg_24h(x), max_1h(x), slope_24h(x), delta_24h(x)) and explicitly state the window length.
9. For discrete metrics: list in the rule expression the set of discrete values that trigger the anomaly (e.g. in {'abnormal','offline','interrupted'}) and state the duration/frequency requirement (e.g. occurs >= N times within a 1h window).
10. Fill in "time_series_requirements" only when the rule needs long observation (trend/persistence/frequency/sliding-window statistics); do not force it for pointwise rules.
11. If "time_series_requirements" is filled in, "rule_expression" must explicitly reflect the time information (window length / duration / frequency threshold), e.g.:
   - avg_24h(x) > T
   - max_6h(x) > T
   - slope_24h(x) > T_slope
   - count_1h(comm_status in {'abnormal','offline'}) >= N
   - duration_6h(x > T) >= 3h
   Expressions like x > T with no time meaning are not allowed.
12. If a pointwise rule requires "consecutive_points > 1", the rule expression must also reflect the consecutive points (e.g. consec_3h(cond)=true); otherwise it is considered non-compliant.
13. Every rule must output both the "pointwise_judgment" and "time_series_requirements" fields; when not applicable they may be written as a default object or empty object {} respectively, but must not be omitted.
14. If any parameter appears in "threshold_parameters", its "parameter_value" must give a **parseable exact number** (e.g. 0.10, 150, 3, 3600); it must not be empty, and must not contain placeholders such as "to be checked/unknown/reference". Prefer a pure numeric string (decimals allowed if needed) and avoid units.
15. **Unified variable names (hard constraint)**: atomic metrics referenced in the rule expression and in "derived_metric_definitions.formula" must use the canonical identifier style that the later metric_catalog will adopt (English snake_case / variable names consistent with the metric semantics, e.g. sf6_gas_pressure, i_coil_steady); inventing engineering aliases (e.g. P_sf6, T_action, PD, energy_status) is **prohibited**. Threshold parameter names may be named separately (e.g. T_ratio_max).
16. Output JSON only, no extra explanation.

[Example Notes]
The following example is only for illustrating the output structure and rule expression style.
The expressions, parameters, and conclusions in the example should not be mechanically copied to other device types.

[Example Input]
Device type:
IPS-3000

Stage-one output JSON:
{
  "device_type": "IPS-3000",
  "instance_fields": ["Device Name", "Device ID"],
  "auxiliary_classification_fields": ["Power Type", "Monitoring Method", "Communication Protocol", "Capacity Class"],
  "common_maintenance_metrics": [
    {
      "metric_name": "Input voltage",
      "metric_definition": "Supply voltage on the device input side, reflecting whether the upstream power supply state is normal."
    },
    {
      "metric_name": "Output voltage",
      "metric_definition": "Supply voltage on the device output side, reflecting the device output power supply state."
    },
    {
      "metric_name": "Communication status",
      "metric_definition": "Status of the communication link between the device and the host system or associated modules."
    }
  ],
  "special_maintenance_metrics": [
    {
      "metric_name": "Battery voltage",
      "metric_definition": "Voltage at the battery pack output terminal, reflecting the backup power supply health.",
      "trigger_conditions": [
        {
          "field_name": "Power Type",
          "field_values": ["AC/DC hybrid", "DC power"]
        }
      ]
    },
    {
      "metric_name": "Remote telemetry status",
      "metric_definition": "Status of the telemetry link between the device and the remote monitoring system.",
      "trigger_conditions": [
        {
          "field_name": "Monitoring Method",
          "field_values": ["Remote monitoring", "Smart monitoring"]
        }
      ]
    }
  ],
  "special_concerns": [
    {
      "concern_description": "For devices configured with remote or smart monitoring, special attention should be paid to unstable communication links, telemetry interruptions, and abnormal remote state reporting.",
      "trigger_conditions": [
        {
          "field_name": "Monitoring Method",
          "field_values": ["Remote monitoring", "Smart monitoring"]
        }
      ]
    },
    {
      "concern_description": "For devices configured with DC power, special attention should be paid to declining backup power capability, battery depletion, and supply stability during power switching.",
      "trigger_conditions": [
        {
          "field_name": "Power Type",
          "field_values": ["AC/DC hybrid", "DC power"]
        }
      ]
    }
  ]
}

[Example Output (showing several types only; the rest indicated by ellipsis)]
{
  "device_type": "IPS-3000",
  "anomaly_rules": [
    {
      "rule_id": "K1",
      "object_name": "Input voltage",
      "object_type": "common_maintenance_metric",
      "applicable_conditions": [],
      "pointwise_judgment": {"single_point_trigger": false, "consecutive_points": 3, "note": "3 consecutive points over threshold"},
      "time_series_requirements": {},
      "input_metrics": ["Input voltage"],
      "derived_metric_definitions": [{"derived_metric_name": "Input voltage deviation rate", "formula": "abs(V_in - V_in_rated) / V_in_rated", "note": "relative deviation from rated"}],
      "judgment_description": "Abnormal when the input voltage deviation rate exceeds the threshold for 3 consecutive points.",
      "rule_expression": "consec_3h(abs(V_in - V_in_rated) / V_in_rated > T_dev) = true",
      "threshold_parameters": [{"parameter_name": "V_in_rated", "meaning": "Rated input voltage (V)", "parameter_value": "220"}, {"parameter_name": "T_dev", "meaning": "Allowed deviation threshold (ratio)", "parameter_value": "0.10"}]
    },
    {
      "rule_id": "K2",
      "object_name": "Concern: unstable remote link / telemetry interruption / abnormal state reporting",
      "object_type": "special_concern",
      "applicable_conditions": [{"field_name": "Monitoring Method", "field_values": ["Remote monitoring", "Smart monitoring"]}],
      "pointwise_judgment": {"single_point_trigger": true, "consecutive_points": 1, "note": "concern-type rules combine multiple metrics"},
      "time_series_requirements": {"requires_long_observation": true, "window_length": "24h", "statistics": ["occurrence_count", "duration"], "missing_data_handling": "not judgeable if missing > 30%"},
      "input_metrics": ["Communication status", "Remote telemetry status"],
      "derived_metric_definitions": [],
      "judgment_description": "If anomalies occur frequently or persist long within a 24h window, the risk of link instability is judged to rise.",
      "rule_expression": "count_24h(comm_status in {'abnormal','offline','interrupted'}) >= N24 OR duration_24h(telemetry_status in {'abnormal','interrupted','unreachable'}) >= T24",
      "threshold_parameters": [{"parameter_name": "N24", "meaning": "Threshold of anomaly count within 24h (times)", "parameter_value": "10"}, {"parameter_name": "T24", "meaning": "Threshold of anomaly duration within 24h (seconds)", "parameter_value": "7200"}]
    },
    ...
  ]
}
"""


derived_metric_stage2_prompt = """
You are an expert in power maintenance fact-checking rule modeling, skilled at turning maintenance metrics and derived metrics into executable, explainable, quantifiable anomaly judgment rules, used to judge whether the data and conclusions in maintenance and fault reports are accurate.

[Task Objective]
The user will provide one [derived metric definition] (from derived_metrics.json) and the catalog information of the atomic metrics it depends on.
Please generate, **for this one derived metric only**, 1–3 high-quality rules usable for stage2.json's "anomaly_rules".
If this derived metric is genuinely unsuitable for building an anomaly judgment rule, you may return an empty list.

[Alignment Requirement with Stage Two of build_maintenance_metrics]
This task uses the same schema and expression conventions as "stage-two anomaly rule generation". The output quality must be consistent with the existing rules for the same device (see [Existing Rule Style Examples]) and must not degrade to vague descriptions or placeholder expressions.

[Quantitative Expression Requirements]
For each rule, output according to the following priority:
1. If it can be directly expressed as a mathematical formula or inequality, output the formula or inequality.
2. If it is a state judgment, output a boolean expression or set-membership judgment expression.
3. If it is a joint judgment over multiple metrics, output a logic-combination expression connected by AND / OR / NOT.

[Input]
Device type:
{device_type}

[Derived Metric Definition to Generate Rules For]
{derived_metric_json}

[Atomic Metric Catalog the Derived Metric Depends On (metric_catalog entries, including unit/normal range, etc.)]
{atomic_metrics_catalog_json}

[Existing Rule Summary (for deduplication; do not generate near-duplicate rules)]
{existing_rules_brief_json}

[Existing Rule Style Examples (imitate their expression style, field completeness, and engineering quality)]
{existing_rules_examples_json}

[Output Format]
Output strict JSON (no Markdown, no explanatory text):

{
  "rules": [
    {
      "object_name": "",
      "object_type": "common_maintenance_metric|special_maintenance_metric|special_concern",
      "applicable_conditions": [
        {
          "field_name": "",
          "field_values": []
        }
      ],
      "pointwise_judgment": {
        "single_point_trigger": true,
        "consecutive_points": 1,
        "note": "Default is hourly sampling; if consecutive_points > 1, the rule expression must explicitly reflect continuity (e.g. consec_3h(cond)=true)"
      },
      "time_series_requirements": {
        "requires_long_observation": true,
        "window_length": "1h|6h|24h|7d",
        "statistics": ["mean|max|min|std|rate_of_change|slope|occurrence_count|duration"]
      },
      "input_metrics": [],
      "derived_metric_definitions": [
        {
          "derived_metric_name": "",
          "formula": "",
          "note": ""
        }
      ],
      "judgment_description": "",
      "rule_expression": "",
      "threshold_parameters": [
        {
          "parameter_name": "",
          "meaning": "",
          "parameter_value": "To be checked in manufacturer manual/nameplate/standard"
        }
      ]
    }
  ]
}

[Field Descriptions]
1. object_name: should use the derived metric's derived_name; if the rule targets the derived metric itself, object_name is that derived metric name.
2. object_type: derived-metric rules usually choose "special_maintenance_metric"; if the derived metric reflects a combined risk over existing common metrics, "special_concern" may be chosen.
3. applicable_conditions: write [] when applicable to all devices; if the derived metric's prerequisites imply a specific configuration restriction, it may be rewritten as field conditions.
4. input_metrics:
   - Prefer directly using the derived metric's derived_name as the sole input (when the rule judges anomalies on the derived quantity itself).
   - If the rule needs to jointly judge multiple atomic metrics, list the indicator_name of the relevant atomic metrics (must be consistent with "metric_name" in the [Atomic Metric Catalog]).
5. derived_metric_definitions:
   - **Must** fill in at least one definition for the current derived metric; the formula should come from the formula / derived_key / atomic_metrics in the input.
   - Atomic metric symbols in the formula and rule expression must use the metric_catalog's "variable_name/db_field_name" (i.e. atomic_metrics.db_field), e.g. circuit_breaker_operating_time, i_coil_steady.
   - Inventing engineering aliases (e.g. T_action, P_sf6, PD) is **prohibited**; threshold parameter names are the exception (e.g. T_ratio_max).
   - Leaving the array empty is prohibited (unless the rule involves no derived computation at all, but this task targets derived metrics, so it should usually not be empty).
6. judgment_description: briefly state "under what conditions it is judged abnormal", consistent with the semantics of the rule expression.
7. rule_expression:
   - Must be executable and explainable; meaningless placeholder variables such as ratio, value, x are prohibited (a true intermediate quantity must be given a formula in derived_metric_definitions).
   - Should explicitly write the derived formula or an equivalent expression, e.g.: circuit_breaker_operating_time / i_coil_steady > T_ratio_max
   - For numeric metrics: prefer time-window statistics (avg_24h, max_6h, slope_24h, delta_24h, etc.) and explicitly state the window length.
   - For discrete metrics: use forms such as in {'abnormal','offline'} and state the duration/frequency requirement.
8. threshold_parameters: every parameter appearing in the expression must be listed; the parameter value must give a **parseable exact number** (e.g. 0.10, 150, 3); it must not be empty or contain "to be determined/unknown/reference".
9. pointwise_judgment and time_series_requirements:
   - Every rule must output these two fields; when not applicable, time_series_requirements may be written as {}.
   - If consecutive_points > 1, the rule expression must reflect consec_Nh(cond)=true.
   - If time_series_requirements is filled in (non-empty), the rule expression must explicitly reflect window/duration/frequency; a bare comparison such as x > T is not allowed.

[Output Requirements]
1. Output only {"rules":[...]}; rules must **not** contain "rule_id" (assigned by the program).
2. Do **not** output a "conclusion" field (the stage2 standard schema does not contain it).
3. Do not add or omit the above schema fields (except rule_id and conclusion).
4. **Unified variable names**: atomic metrics in the rule expression / derived formula must use the "variable_name" or "db_field_name" from the [Atomic Metric Catalog]; even if the [Existing Rule Style Examples] contain historical aliases such as P_sf6 or T_action, they must not be copied.
5. The derived metric's formula must be fully presented in "derived_metric_definitions", and the db_field expression must be used directly in "rule_expression" (or derived_key, if it is a legal identifier).
6. Thresholds should be given reasonable values combining the derived metric's output_unit, the atomic metrics' normal ranges, and engineering experience; obviously unreasonable dimensions must not be fabricated.
7. Try to avoid near-duplicate "object_name + rule_expression" with the [Existing Rule Summary].
8. If there is no suitable anomaly rule, return {"rules": []}.
9. Output JSON only, no extra explanation.

[Derived-Metric Rule Example (only illustrates the structure and expression style; do not mechanically copy the values)]
Input derived metric:
{
  "derived_name": "Operating time to current ratio",
  "derived_key": "operating_time_to_coil_current_ratio",
  "formula": "circuit_breaker_operating_time / i_coil_steady",
  "atomic_metrics": [
    {"db_field": "circuit_breaker_operating_time", "indicator_name": "Circuit breaker operating time", "unit": "ms"},
    {"db_field": "i_coil_steady", "indicator_name": "Circuit breaker open/close coil current", "unit": "A"}
  ],
  "output_unit": "ms/A"
}

Qualified output example:
{
  "rules": [
    {
      "object_name": "Operating time to current ratio",
      "object_type": "special_maintenance_metric",
      "applicable_conditions": [],
      "pointwise_judgment": {"single_point_trigger": true, "consecutive_points": 1, "note": "compute the derived ratio and judge anomaly within a single open/close operation event"},
      "time_series_requirements": {},
      "input_metrics": ["Operating time to current ratio"],
      "derived_metric_definitions": [
        {
          "derived_metric_name": "Operating time to current ratio",
          "formula": "circuit_breaker_operating_time / i_coil_steady",
          "note": "Ratio of circuit breaker open/close operating time to coil steady-state current, reflecting electromagnetic drive efficiency and mechanical resistance"
        }
      ],
      "judgment_description": "In a single open/close operation, if the operating time to current ratio exceeds the upper threshold, it indicates a risk of mechanical jamming or coil aging.",
      "rule_expression": "circuit_breaker_operating_time / i_coil_steady > T_ratio_max OR circuit_breaker_operating_time / i_coil_steady < T_ratio_min",
      "threshold_parameters": [
        {"parameter_name": "T_ratio_min", "meaning": "Lower threshold of operating time to current ratio (ms/A)", "parameter_value": "8.0"},
        {"parameter_name": "T_ratio_max", "meaning": "Upper threshold of operating time to current ratio (ms/A)", "parameter_value": "25.0"}
      ]
    }
  ]
}
"""
