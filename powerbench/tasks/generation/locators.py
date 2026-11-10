from __future__ import annotations

import itertools
import re
from collections import defaultdict
from collections.abc import Iterable

from ..models import Device, Locator

_FORBIDDEN_KEYS = {
    "Archive ID",
    "Device ID",
    "Device name",
    "Tag number",
    "Device tag",
    "Name",
    "Factory serial",
    "Contract No.",
}
_DISFAVORED_KEYS = {
    "Installation location",
    "Panel ID",
    "Bay ID",
    "Cabinet No.",
}
_PREFERRED_KEYS = (
    "Project",
    "Station / line",
    "Station",
    "Equipment class",
    "Equipment level",
    "Device type",
    "Primary/standby",
    "Phase",
    "System position",
    "System architecture",
    "Communication protocol",
    "Insulation type",
    "Manufacturer",
    "Battery type",
    "Charging method",
    "Input voltage",
    "Output current",
    "Voltage class",
)
_KEY_GROUPS = {
    "Project": "site",
    "Station / line": "site",
    "Station": "site",
    "Equipment class": "equipment_class",
    "Equipment level": "equipment_class",
    "Device type": "equipment_type",
    "Installation location": "physical_location",
    "System position": "system_location",
}


def metadata_key_group(key: str) -> str:
    """Return the semantic dimension represented by one metadata key."""

    return _KEY_GROUPS.get(key, key)


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()


def locator_candidates(device: Device) -> list[tuple[str, str]]:
    values: list[tuple[str, str]] = []
    preferred_rank = {key: index for index, key in enumerate(_PREFERRED_KEYS)}
    for key, raw_value in device.metadata.items():
        value = _clean(raw_value)
        if not value or key in _FORBIDDEN_KEYS:
            continue
        if len(value) > 80 or value in {"-", "n/a", "unknown"}:
            continue
        values.append((key, value))
    values.sort(
        key=lambda item: (
            item[0] in _DISFAVORED_KEYS,
            preferred_rank.get(item[0], len(preferred_rank) + 1),
            item[0],
            item[1],
        )
    )
    return values


def build_strong_locators(
    devices: Iterable[Device], *, max_fields: int | None = None
) -> dict[str, Locator]:
    """Find globally unique 3--4-field locators with ablation ambiguity.

    A locator is accepted only when its full conjunction selects one device and
    deleting any one clue leaves at least two candidates. This makes the
    Indirect condition genuinely conjunctive rather than a disguised ID lookup.
    """

    if max_fields is not None and max_fields < 4:
        raise ValueError("max_fields must be at least four when provided")
    device_list = sorted(devices, key=lambda item: item.uid)
    inverted: dict[tuple[str, str], set[str]] = defaultdict(set)
    values_by_uid: dict[str, list[tuple[str, str]]] = {}
    for device in device_list:
        pairs = locator_candidates(device)
        values_by_uid[device.uid] = pairs
        for pair in pairs:
            inverted[pair].add(device.uid)

    result: dict[str, Locator] = {}
    for device in device_list:
        pairs = values_by_uid[device.uid]
        # A clue that is already unique defeats the point of Indirect Anchor.
        pairs = [pair for pair in pairs if len(inverted[pair]) >= 2]
        for width in (3, 4):
            if len(pairs) < width:
                continue
            searchable = pairs[:max_fields] if max_fields is not None else pairs
            for combo in itertools.combinations(searchable, width):
                semantic_groups = [metadata_key_group(key) for key, _ in combo]
                if len(set(semantic_groups)) != len(semantic_groups):
                    continue
                full = set.intersection(*(inverted[pair] for pair in combo))
                if full != {device.uid}:
                    continue
                confusers: set[str] = set()
                valid = True
                for omitted in range(width):
                    remainder = [combo[index] for index in range(width) if index != omitted]
                    matches = set.intersection(*(inverted[pair] for pair in remainder))
                    matches.discard(device.uid)
                    if not matches:
                        valid = False
                        break
                    confusers.add(sorted(matches)[0])
                if valid:
                    result[device.uid] = Locator(
                        fields=tuple(combo),
                        ablation_confusers=tuple(sorted(confusers)),
                    )
                    break
            if device.uid in result:
                break
    return result


def render_locator(locator: Locator) -> str:
    return ", ".join(f'{key} is "{value}"' for key, value in locator.fields)


def matching_devices(locator: Locator, devices: Iterable[Device]) -> list[str]:
    expected = dict(locator.fields)
    return sorted(
        device.uid
        for device in devices
        if all(_clean(device.metadata.get(key, "")) == value for key, value in expected.items())
    )