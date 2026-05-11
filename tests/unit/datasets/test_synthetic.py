"""Unit tests for synthetic dataset generation (template-based)."""

from sarathi_agent_inspect.datasets.synthetic import EDGE_CASE_TEMPLATES, SyntheticGenerator

# ── Template Edge-Case Tests ────────────────────────────────────────


def test_generate_from_templates_all_categories():
    """Test generating from all built-in categories."""
    records = SyntheticGenerator.generate_from_templates()

    assert len(records) > 0

    # Every record should have synthetic metadata
    for record in records:
        assert record["metadata"]["synthetic"] is True
        assert record["metadata"]["type"] == "template_edge_case"
        assert "category" in record["metadata"]


def test_generate_from_templates_specific_category():
    """Test generating from a single category."""
    records = SyntheticGenerator.generate_from_templates(categories=["unicode"])

    assert len(records) == len(EDGE_CASE_TEMPLATES["unicode"])
    for record in records:
        assert record["metadata"]["category"] == "unicode"


def test_generate_from_templates_multiple_categories():
    """Test generating from multiple specific categories."""
    records = SyntheticGenerator.generate_from_templates(categories=["empty_input", "numeric"])

    expected_count = len(EDGE_CASE_TEMPLATES["empty_input"]) + len(EDGE_CASE_TEMPLATES["numeric"])
    assert len(records) == expected_count


def test_generate_from_templates_with_base_record():
    """Test template generation with a base record overlay."""
    base = {"expected_output": "I cannot help with that.", "category": "safety"}

    records = SyntheticGenerator.generate_from_templates(
        categories=["injection"],
        base_record=base,
    )

    assert len(records) > 0
    for record in records:
        # Base record fields should be present
        assert record["expected_output"] == "I cannot help with that."
        # Template fields should override where applicable
        assert "input" in record
        assert record["metadata"]["synthetic"] is True


def test_generate_from_templates_empty_category():
    """Test generating from a nonexistent category returns empty list."""
    records = SyntheticGenerator.generate_from_templates(categories=["nonexistent"])
    assert records == []


def test_edge_case_templates_structure():
    """Test that all template categories exist and have valid records."""
    expected_categories = [
        "empty_input",
        "unicode",
        "injection",
        "max_length",
        "special_characters",
        "numeric",
        "adversarial",
    ]

    for category in expected_categories:
        assert category in EDGE_CASE_TEMPLATES, f"Missing template category: {category}"
        templates = EDGE_CASE_TEMPLATES[category]
        assert len(templates) > 0, f"Empty template category: {category}"

        for template in templates:
            assert "input" in template, f"Template missing 'input' in {category}"
            assert "metadata" in template, f"Template missing 'metadata' in {category}"


def test_injection_templates_coverage():
    """Test that injection templates cover key attack vectors."""
    injection_records = SyntheticGenerator.generate_from_templates(categories=["injection"])

    edge_case_types = {r["metadata"]["edge_case"] for r in injection_records}
    assert "prompt_injection_basic" in edge_case_types
    assert "sql_injection" in edge_case_types
    assert "xss_attempt" in edge_case_types
