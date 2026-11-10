stage_1_prompt = """
## I. Two-Stage Prompt Design (Revised)

### Stage 1: Identify subdivision metric dimensions

Input: device type path
Output: which subdivision metric dimensions this type of device has


## Role
You are an expert in power system equipment ledgers and O&M management, familiar with the equipment classification, engineering configuration, and instantiation naming rules of converter transformers in UHV DC projects.

## Task
The user will provide a device type path. Please analyze the fields involved in ledger construction for this type of device, and split the fields into two categories:

1. **Distinguishing Dimensions**: dimensions used to distinguish different subtypes or technical configurations of this type of device, dimensions that cause changes in technical configuration, monitoring focus, or diagnostic logic
   For example: voltage level, connection method, cooling method, bushing type, grid-side voltage, structural form

2. **Instantiation Dimensions**: dimensions used to generate concrete device instances and form ledger records
   For example: belonging project, converter station, system location, phase, whether standby, device number

Note:
- **Distinguishing Dimensions** emphasize "technical classification"
- **Instantiation Dimensions** emphasize "instance attributes after the device is mapped to a concrete project and concrete location"
- Do not mistake pure naming information for technical classification dimensions
- Only keep fields that are truly meaningful for this device type and commonly used for equipment management and instantiation record-keeping

## Example Input
Device type path: Primary Equipment (main body of power transmission) -> UHV Backbone Grid Equipment -> Converter Transformer -> +/-800kV High-end Converter Transformer

## Example Output
{
  "Device Type": "+/-800kV High-end Converter Transformer",
  "Distinguishing Dimensions": [
    {
      "Dimension Name": "Connection Method",
      "Dimension Description": "Technical dimension used to distinguish the winding connection configuration of this type of converter transformer",
      "Recommended Values": ["Y/Y", "Y/delta"],
      "Required": true
    },
    {
      "Dimension Name": "Cooling Method",
      "Dimension Description": "Used to distinguish the cooling configuration adopted by the device",
      "Recommended Values": ["Forced oil circulation air cooling", "Forced oil circulation water cooling"],
      "Required": true
    },
    {
      "Dimension Name": "Bushing Type",
      "Dimension Description": "Used to distinguish the insulation form of the valve-side bushing",
      "Recommended Values": ["Air-insulated bushing", "SF6 gas-insulated bushing"],
      "Required": true
    },
    {
      "Dimension Name": "Grid-side Voltage",
      "Dimension Description": "Used to distinguish the AC-side connection voltage level",
      "Recommended Values": ["500kV", "750kV"],
      "Required": true
    },
    {
      "Dimension Name": "Voltage Regulation Method",
      "Dimension Description": "Used to distinguish the on-load/off-load voltage regulation configuration of this type of converter transformer",
      "Recommended Values": ["On-load voltage regulation", "Off-load voltage regulation"],
      "Required": false
    }
  ],
  "Instantiation Dimensions": [
    {
      "Dimension Name": "Belonging Project",
      "Dimension Description": "The DC project or converter station the device belongs to",
      "Recommended Values": ["Jinping Converter Station", "Xiangjiaba Converter Station", "Xiluodu Converter Station", "Fufeng DC Project"],
      "Required": true
    },
    {
      "Dimension Name": "System Location",
      "Dimension Description": "The pole position or system location of the device in the project",
      "Recommended Values": ["Pole 1 System", "Pole 2 System"],
      "Required": true
    },
    {
      "Dimension Name": "Phase",
      "Dimension Description": "The phase the device belongs to",
      "Recommended Values": ["Phase A", "Phase B", "Phase C"],
      "Required": true
    },
    {
      "Dimension Name": "Unit Number",
      "Dimension Description": "Distinguished by unit/set number within the same converter station (if applicable)",
      "Recommended Values": ["Unit 1", "Unit 2"],
      "Required": false
    },
    {
      "Dimension Name": "Primary/Standby Attribute",
      "Dimension Description": "The primary/standby attribute of the same type of device (if the project has a primary/standby configuration)",
      "Recommended Values": ["Primary device", "Standby device"],
      "Required": false
    }
  ],
  "Notes": {
    "Which dimensions are used for technical classification": ["Connection Method", "Cooling Method", "Bushing Type", "Grid-side Voltage"],
    "Which dimensions are used to generate instance ledgers": ["Belonging Project", "System Location", "Phase", "Unit Number", "Primary/Standby Attribute"]
  }
}

## User Input
Device type path: {device_type_path}

## Output Requirements
Please output JSON in the following format:

{
  "Device Type": "",
  "Distinguishing Dimensions": [
    {
      "Dimension Name": "",
      "Dimension Description": "",
      "Recommended Values": [],
      "Required": true
    }
  ],
  "Instantiation Dimensions": [
    {
      "Dimension Name": "",
      "Dimension Description": "",
      "Recommended Values": [],
      "Required": true
    }
  ],
  "Notes": {
    "Which dimensions are used for technical classification": [],
    "Which dimensions are used to generate instance ledgers": []
  }
}

## Constraints
1. Only output dimensions highly relevant to this device type; do not generalize to fields common to all transformers
2. Recommended values should be as close as possible to real engineering expressions
3. If a dimension exists but its values should not be forcibly enumerated, use more conservative and common values
4. If you cannot confirm with high confidence whether a dimension applies, do not output that dimension
5. Only output JSON, do not output explanatory text

"""
stage_2_prompt = """
### Stage 2: Generate equipment list based on metric combinations

Input: the subdivision metric dimensions from Stage 1
Output: equipment list (each device contains the device name and the attribute values of each metric)


## Role
You are a power system equipment management expert, familiar with the engineering configuration and ledger management of UHV equipment.

## Task
Based on the dimensions and their value ranges identified in Stage 1, generate the equipment list for this device type.

Please note:
- The attribute fields in the equipment list must strictly come from the dimension names in the input
- Do not add fields that do not exist in the input on your own
- If a dimension does not appear in the input, do not add that field in the output
- Do not mechanically enumerate all combinations; only generate instances that are reasonable and self-consistent in real engineering

## Example Input
{
  "Device Type": "+/-800kV High-end Converter Transformer",
  "Dimension List": [
    {
      "Dimension Name": "Belonging Project",
      "Dimension Description": "The DC project or converter station the device belongs to",
      "Value Range": ["Jinping Converter Station"]
    },
    {
      "Dimension Name": "System Location",
      "Dimension Description": "The pole position or system location of the device in the project",
      "Value Range": ["Pole 1 System", "Pole 2 System"]
    },
    {
      "Dimension Name": "Phase",
      "Dimension Description": "The phase the device belongs to",
      "Value Range": ["Phase A", "Phase B", "Phase C"]
    },
    {
      "Dimension Name": "Cooling Method",
      "Dimension Description": "The cooling configuration adopted by the device",
      "Value Range": ["Forced oil circulation water cooling"]
    },
    {
      "Dimension Name": "Connection Method",
      "Dimension Description": "Winding connection configuration",
      "Value Range": ["Y/Y"]
    },
    {
      "Dimension Name": "Bushing Type",
      "Dimension Description": "Valve-side bushing insulation form",
      "Value Range": ["SF6 gas-insulated bushing"]
    }
  ]
}

## Example Output
{
  "Device Type": "+/-800kV High-end Converter Transformer",
  "Combination Constraints": [
    "Belonging Project fixed as Jinping Converter Station",
    "System Location expanded by Pole 1 System and Pole 2 System",
    "Under each system location expanded by Phase A, Phase B, Phase C",
    "Cooling Method fixed as Forced oil circulation water cooling under this input condition",
    "Connection Method fixed as Y/Y",
    "Bushing Type fixed as SF6 gas-insulated bushing"
  ],
  "Naming Component Fields": ["Belonging Project", "System Location", "Device Type", "Phase"],
  "Naming Rule": "{Belonging Project}{System Location}{Device Type}{Phase}",
  "Equipment List Fields": ["Device Name", "Belonging Project", "System Location", "Phase", "Cooling Method", "Connection Method", "Bushing Type"],
  "Equipment List": [
    {
      "Device Name": "Jinping Converter Station Pole 1 System +/-800kV High-end Converter Transformer Phase A",
      "Belonging Project": "Jinping Converter Station",
      "System Location": "Pole 1 System",
      "Phase": "Phase A",
      "Cooling Method": "Forced oil circulation water cooling",
      "Connection Method": "Y/Y",
      "Bushing Type": "SF6 gas-insulated bushing"
    },
    {
      "Device Name": "Jinping Converter Station Pole 1 System +/-800kV High-end Converter Transformer Phase B",
      "Belonging Project": "Jinping Converter Station",
      "System Location": "Pole 1 System",
      "Phase": "Phase B",
      "Cooling Method": "Forced oil circulation water cooling",
      "Connection Method": "Y/Y",
      "Bushing Type": "SF6 gas-insulated bushing"
    },
    {
      "Device Name": "Jinping Converter Station Pole 1 System +/-800kV High-end Converter Transformer Phase C",
      "Belonging Project": "Jinping Converter Station",
      "System Location": "Pole 1 System",
      "Phase": "Phase C",
      "Cooling Method": "Forced oil circulation water cooling",
      "Connection Method": "Y/Y",
      "Bushing Type": "SF6 gas-insulated bushing"
    },
    {
      "Device Name": "Jinping Converter Station Pole 2 System +/-800kV High-end Converter Transformer Phase A",
      "Belonging Project": "Jinping Converter Station",
      "System Location": "Pole 2 System",
      "Phase": "Phase A",
      "Cooling Method": "Forced oil circulation water cooling",
      "Connection Method": "Y/Y",
      "Bushing Type": "SF6 gas-insulated bushing"
    },
    {
      "Device Name": "Jinping Converter Station Pole 2 System +/-800kV High-end Converter Transformer Phase B",
      "Belonging Project": "Jinping Converter Station",
      "System Location": "Pole 2 System",
      "Phase": "Phase B",
      "Cooling Method": "Forced oil circulation water cooling",
      "Connection Method": "Y/Y",
      "Bushing Type": "SF6 gas-insulated bushing"
    },
    {
      "Device Name": "Jinping Converter Station Pole 2 System +/-800kV High-end Converter Transformer Phase C",
      "Belonging Project": "Jinping Converter Station",
      "System Location": "Pole 2 System",
      "Phase": "Phase C",
      "Cooling Method": "Forced oil circulation water cooling",
      "Connection Method": "Y/Y",
      "Bushing Type": "SF6 gas-insulated bushing"
    }
  ]
}

## User Input
{
  "Device Type": "{device_type}",
  "Dimension List": {dimension_list}
}

## Output Format (JSON)
{
  "Device Type": "",
  "Combination Constraints": [],
  "Naming Component Fields": [],
  "Naming Rule": "",
  "Equipment List Fields": [],
  "Equipment List": [
    {}
  ]
}

## Requirements
1. "Equipment List Fields" must consist of "Device Name" plus all dimension names in the input
2. The fields of each record in "Equipment List" must be exactly consistent with "Equipment List Fields"
3. Do not output dimensions that do not appear in the input
4. If there are fixed pairing relationships between certain dimensions, first clarify them in "Combination Constraints", then generate the equipment list
5. The device name can only be constructed using the fields and values already present in the input, plus the device type itself
6. Do not mechanically enumerate all combinations; only generate instances that are reasonable and self-consistent in real engineering
7. Only output JSON, do not output extra explanation

"""
