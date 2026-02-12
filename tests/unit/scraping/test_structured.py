"""Unit tests for research_agent.scraping.structured - JSON-LD/Schema.org extraction."""

from __future__ import annotations

import json

import pytest

from research_agent.scraping.structured import (
    StructuredDataExtractor,
    StructuredDataItem,
    StructuredDataResult,
)

# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


class TestStructuredDataItem:
    """StructuredDataItem Pydantic model validation."""

    def test_default_construction(self) -> None:
        item = StructuredDataItem()
        assert item.schema_type == "Unknown"
        assert item.properties == {}
        assert item.source == "json_ld"

    def test_custom_construction(self) -> None:
        item = StructuredDataItem(
            schema_type="Article",
            properties={"headline": "Test Article"},
            source="json_ld",
        )
        assert item.schema_type == "Article"
        assert item.properties["headline"] == "Test Article"


class TestStructuredDataResult:
    """StructuredDataResult Pydantic model validation."""

    def test_default_result(self) -> None:
        result = StructuredDataResult()
        assert result.items == []
        assert result.has_structured_data is False
        assert result.schema_types == []
        assert result.quality_boost == 0.0

    def test_quality_boost_upper_bound(self) -> None:
        with pytest.raises(ValueError):
            StructuredDataResult(quality_boost=0.3)


# ---------------------------------------------------------------------------
# Extractor tests
# ---------------------------------------------------------------------------


def _make_json_ld(data: dict | list) -> str:
    """Helper: wrap JSON-LD data in a script tag."""
    return f'<script type="application/ld+json">{json.dumps(data)}</script>'


class TestStructuredDataExtractor:
    """StructuredDataExtractor JSON-LD parsing and property extraction."""

    def test_default_init(self) -> None:
        extractor = StructuredDataExtractor()
        assert extractor.max_items == 10

    def test_custom_max_items(self) -> None:
        extractor = StructuredDataExtractor(max_items=5)
        assert extractor.max_items == 5

    def test_empty_html_returns_empty(self) -> None:
        extractor = StructuredDataExtractor()
        result = extractor.extract("")
        assert result.has_structured_data is False

    def test_no_json_ld_returns_empty(self) -> None:
        extractor = StructuredDataExtractor()
        html = "<html><body><p>No structured data.</p></body></html>"
        result = extractor.extract(html)
        assert result.has_structured_data is False

    def test_article_extraction(self) -> None:
        extractor = StructuredDataExtractor()
        data = {
            "@type": "Article",
            "headline": "Understanding RAG",
            "description": "An overview of retrieval-augmented generation.",
            "author": {"@type": "Person", "name": "Jane Doe"},
            "datePublished": "2024-06-15",
        }
        html = _make_json_ld(data)
        result = extractor.extract(html)

        assert result.has_structured_data is True
        assert "Article" in result.schema_types
        assert len(result.items) == 1
        assert result.items[0].properties["headline"] == "Understanding RAG"
        assert result.items[0].properties["author"] == "Jane Doe"

    def test_product_extraction(self) -> None:
        extractor = StructuredDataExtractor()
        data = {
            "@type": "Product",
            "name": "Widget Pro",
            "description": "A professional widget.",
            "brand": {"@type": "Brand", "name": "WidgetCo"},
            "sku": "WP-001",
        }
        html = _make_json_ld(data)
        result = extractor.extract(html)

        assert result.has_structured_data is True
        assert "Product" in result.schema_types
        assert result.items[0].properties["name"] == "Widget Pro"
        assert result.items[0].properties["brand"] == "WidgetCo"

    def test_news_article_extraction(self) -> None:
        extractor = StructuredDataExtractor()
        data = {
            "@type": "NewsArticle",
            "headline": "Breaking News",
            "datePublished": "2024-06-15",
            "publisher": {"@type": "Organization", "name": "News Corp"},
        }
        html = _make_json_ld(data)
        result = extractor.extract(html)

        assert "NewsArticle" in result.schema_types
        assert result.items[0].properties["publisher"] == "News Corp"

    def test_graph_array_extraction(self) -> None:
        extractor = StructuredDataExtractor()
        data = {
            "@graph": [
                {
                    "@type": "WebPage",
                    "name": "My Page",
                    "url": "https://example.com",
                },
                {
                    "@type": "Article",
                    "headline": "Article in Graph",
                    "datePublished": "2024-01-01",
                },
            ]
        }
        html = _make_json_ld(data)
        result = extractor.extract(html)

        assert len(result.items) == 2
        types = {item.schema_type for item in result.items}
        assert "WebPage" in types
        assert "Article" in types

    def test_top_level_array_extraction(self) -> None:
        extractor = StructuredDataExtractor()
        data = [
            {"@type": "Person", "name": "Alice"},
            {"@type": "Person", "name": "Bob"},
        ]
        html = _make_json_ld(data)
        result = extractor.extract(html)

        assert len(result.items) == 2

    def test_multiple_json_ld_blocks(self) -> None:
        extractor = StructuredDataExtractor()
        block1 = _make_json_ld({"@type": "Article", "headline": "First"})
        block2 = _make_json_ld({"@type": "Organization", "name": "Acme"})
        html = f"<html>{block1}{block2}</html>"
        result = extractor.extract(html)

        assert len(result.items) == 2
        assert {"Article", "Organization"} == set(result.schema_types)

    def test_invalid_json_skipped(self) -> None:
        extractor = StructuredDataExtractor()
        html = '<script type="application/ld+json">{invalid json}</script>'
        result = extractor.extract(html)
        assert result.has_structured_data is False

    def test_item_without_type_skipped(self) -> None:
        extractor = StructuredDataExtractor()
        data = {"headline": "No Type Here"}
        html = _make_json_ld(data)
        result = extractor.extract(html)
        assert result.has_structured_data is False

    def test_item_without_properties_skipped(self) -> None:
        extractor = StructuredDataExtractor()
        # "SomeRandom" type has no matching props in PROPERTY_MAP
        # and the generic extraction (name, description, url, datePublished) finds nothing
        data = {"@type": "SomeRandom", "irrelevant_field": "value"}
        html = _make_json_ld(data)
        result = extractor.extract(html)
        assert result.has_structured_data is False

    def test_unknown_type_uses_generic_properties(self) -> None:
        extractor = StructuredDataExtractor()
        data = {
            "@type": "CustomThing",
            "name": "My Custom Thing",
            "description": "A custom schema type.",
        }
        html = _make_json_ld(data)
        result = extractor.extract(html)

        assert result.has_structured_data is True
        assert result.items[0].properties["name"] == "My Custom Thing"

    def test_type_as_list_uses_first(self) -> None:
        extractor = StructuredDataExtractor()
        data = {
            "@type": ["Article", "NewsArticle"],
            "headline": "Multi-type Article",
        }
        html = _make_json_ld(data)
        result = extractor.extract(html)

        assert result.items[0].schema_type == "Article"

    def test_max_items_truncation(self) -> None:
        extractor = StructuredDataExtractor(max_items=2)
        items = [{"@type": "Person", "name": f"Person {i}"} for i in range(5)]
        html = _make_json_ld(items)
        result = extractor.extract(html)
        assert len(result.items) <= 2

    def test_nested_value_simplification(self) -> None:
        extractor = StructuredDataExtractor()
        data = {
            "@type": "Article",
            "headline": "Test",
            "author": {"@type": "Person", "name": "John Smith"},
            "publisher": {"@type": "Organization", "name": "Publisher Corp"},
        }
        html = _make_json_ld(data)
        result = extractor.extract(html)

        # Nested objects should be simplified to their "name" field
        assert result.items[0].properties["author"] == "John Smith"
        assert result.items[0].properties["publisher"] == "Publisher Corp"

    def test_list_values_simplified(self) -> None:
        extractor = StructuredDataExtractor()
        data = {
            "@type": "Person",
            "name": "Alice",
            "sameAs": ["https://twitter.com/alice", "https://github.com/alice"],
        }
        html = _make_json_ld(data)
        result = extractor.extract(html)

        same_as = result.items[0].properties["sameAs"]
        assert isinstance(same_as, list)
        assert len(same_as) == 2


# ---------------------------------------------------------------------------
# Quality boost tests
# ---------------------------------------------------------------------------


class TestQualityBoost:
    """StructuredDataExtractor quality boost calculation."""

    def test_no_items_zero_boost(self) -> None:
        extractor = StructuredDataExtractor()
        html = "<html><body>No data</body></html>"
        result = extractor.extract(html)
        assert result.quality_boost == 0.0

    def test_article_boost(self) -> None:
        extractor = StructuredDataExtractor()
        data = {"@type": "Article", "headline": "Test"}
        html = _make_json_ld(data)
        result = extractor.extract(html)
        assert result.quality_boost == 0.1  # Article boost

    def test_faq_page_highest_boost(self) -> None:
        extractor = StructuredDataExtractor()
        data = {"@type": "FAQPage", "name": "FAQ", "description": "Questions"}
        html = _make_json_ld(data)
        result = extractor.extract(html)
        assert result.quality_boost == 0.15  # FAQPage boost

    def test_multiple_items_get_bonus(self) -> None:
        extractor = StructuredDataExtractor()
        data = [
            {"@type": "Article", "headline": "A"},
            {"@type": "Article", "headline": "B"},
            {"@type": "Article", "headline": "C"},
        ]
        html = _make_json_ld(data)
        result = extractor.extract(html)
        # Article (0.10) + multi bonus (2 * 0.01 = 0.02) = 0.12
        assert result.quality_boost > 0.10

    def test_boost_capped_at_0_2(self) -> None:
        extractor = StructuredDataExtractor()
        # Many items to push bonus high
        data = [{"@type": "FAQPage", "name": f"FAQ {i}"} for i in range(10)]
        html = _make_json_ld(data)
        result = extractor.extract(html)
        assert result.quality_boost <= 0.2


# ---------------------------------------------------------------------------
# Content formatting tests
# ---------------------------------------------------------------------------


class TestFormatForContent:
    """StructuredDataExtractor.format_for_content output."""

    def test_empty_result_returns_empty_string(self) -> None:
        extractor = StructuredDataExtractor()
        result = StructuredDataResult()
        assert extractor.format_for_content(result) == ""

    def test_formats_article_data(self) -> None:
        extractor = StructuredDataExtractor()
        data = {
            "@type": "Article",
            "headline": "Test Article",
            "datePublished": "2024-06-15",
        }
        html = _make_json_ld(data)
        result = extractor.extract(html)
        formatted = extractor.format_for_content(result)

        assert "[STRUCTURED DATA]" in formatted
        assert "Type: Article" in formatted
        assert "headline: Test Article" in formatted

    def test_formats_multiple_items(self) -> None:
        extractor = StructuredDataExtractor()
        data = [
            {"@type": "Person", "name": "Alice"},
            {"@type": "Person", "name": "Bob"},
        ]
        html = _make_json_ld(data)
        result = extractor.extract(html)
        formatted = extractor.format_for_content(result)

        assert formatted.count("Type: Person") == 2
        assert "Alice" in formatted
        assert "Bob" in formatted


# ---------------------------------------------------------------------------
# Branch coverage: _extract_json_ld edge cases
# ---------------------------------------------------------------------------


class TestExtractJsonLdBranches:
    """Edge cases in _extract_json_ld for uncovered branch paths."""

    def test_graph_with_non_list_value(self) -> None:
        """When @graph contains a non-list value, the branch skips graph
        iteration and continues to the next block (line 268->273)."""
        extractor = StructuredDataExtractor()
        data = {"@graph": "not-a-list"}
        html = _make_json_ld(data)
        result = extractor.extract(html)
        assert result.has_structured_data is False

    def test_graph_items_where_parse_returns_none(self) -> None:
        """When @graph contains items that _parse_json_ld_item returns None
        for (e.g. items without @type), those are filtered out (lines 271->269)."""
        extractor = StructuredDataExtractor()
        data = {
            "@graph": [
                {"no_type": "just data"},  # No @type, returns None
                {"@type": "Article", "headline": "Valid Article"},
            ]
        }
        html = _make_json_ld(data)
        result = extractor.extract(html)

        assert result.has_structured_data is True
        assert len(result.items) == 1
        assert result.items[0].schema_type == "Article"

    def test_graph_all_items_return_none(self) -> None:
        """When all @graph items fail parsing, the result is empty."""
        extractor = StructuredDataExtractor()
        data = {
            "@graph": [
                {"no_type": "item1"},
                {"no_type": "item2"},
            ]
        }
        html = _make_json_ld(data)
        result = extractor.extract(html)
        assert result.has_structured_data is False

    def test_top_level_list_with_some_none_items(self) -> None:
        """When a top-level list has items that parse to None, only valid
        items are included (lines 279->277)."""
        extractor = StructuredDataExtractor()
        data = [
            {"@type": "Person", "name": "Alice"},
            {"no_type": "invalid"},  # No @type, returns None
            {"@type": "Person", "name": "Charlie"},
        ]
        html = _make_json_ld(data)
        result = extractor.extract(html)

        assert len(result.items) == 2

    def test_single_dict_parse_returns_none(self) -> None:
        """When a top-level dict without @type is parsed, _parse_json_ld_item
        returns None and no item is added (line 281->259)."""
        extractor = StructuredDataExtractor()
        data = {"headline": "No type field here"}
        html = _make_json_ld(data)
        result = extractor.extract(html)
        assert result.has_structured_data is False


# ---------------------------------------------------------------------------
# Branch coverage: _parse_json_ld_item edge cases
# ---------------------------------------------------------------------------


class TestParseJsonLdItemBranches:
    """Edge cases in _parse_json_ld_item for uncovered paths."""

    def test_non_dict_input_returns_none(self) -> None:
        """When _parse_json_ld_item receives a non-dict (e.g. a string),
        it returns None (line 298)."""
        extractor = StructuredDataExtractor()
        result = extractor._parse_json_ld_item("not a dict")
        assert result is None

    def test_integer_input_returns_none(self) -> None:
        """Non-dict primitive types also return None."""
        extractor = StructuredDataExtractor()
        result = extractor._parse_json_ld_item(42)
        assert result is None

    def test_list_input_returns_none(self) -> None:
        """A list input (not a dict) returns None."""
        extractor = StructuredDataExtractor()
        result = extractor._parse_json_ld_item(["@type", "Article"])
        assert result is None


# ---------------------------------------------------------------------------
# Branch coverage: _simplify_value edge cases
# ---------------------------------------------------------------------------


class TestSimplifyValueBranches:
    """Edge cases in _simplify_value for uncovered paths."""

    def test_dict_with_only_at_type(self) -> None:
        """When a dict has none of the expected keys (name, @value, value, text)
        but has @type, the @type is returned as a string (line 343)."""
        extractor = StructuredDataExtractor()
        result = extractor._simplify_value({"@type": "ImageObject", "url": "http://img.png"})
        assert result == "ImageObject"

    def test_dict_with_no_expected_keys_no_type(self) -> None:
        """When a dict has none of the expected keys and no @type,
        str(value) of the dict is returned (line 343 via value.get returning dict)."""
        extractor = StructuredDataExtractor()
        result = extractor._simplify_value({"unknown": "data"})
        # Falls through to str(value.get("@type", value)), @type missing so returns str(dict)
        assert isinstance(result, str)

    def test_dict_with_at_value_key(self) -> None:
        """When a dict has '@value', that value is extracted."""
        extractor = StructuredDataExtractor()
        result = extractor._simplify_value({"@value": "2024-01-15", "@type": "Date"})
        assert result == "2024-01-15"

    def test_dict_with_value_key(self) -> None:
        """When a dict has 'value', that is extracted."""
        extractor = StructuredDataExtractor()
        result = extractor._simplify_value({"value": "100", "@type": "MonetaryAmount"})
        assert result == "100"

    def test_dict_with_text_key(self) -> None:
        """When a dict has 'text', that is extracted."""
        extractor = StructuredDataExtractor()
        result = extractor._simplify_value({"text": "Hello world", "@type": "TextObject"})
        assert result == "Hello world"


# ---------------------------------------------------------------------------
# Branch coverage: _format_value edge cases
# ---------------------------------------------------------------------------


class TestFormatValueBranches:
    """Edge cases in _format_value for uncovered paths."""

    def test_empty_list_returns_empty_string(self) -> None:
        """An empty list should return an empty string (lines 360-361)."""
        extractor = StructuredDataExtractor()
        result = extractor._format_value([])
        assert result == ""

    def test_list_with_only_falsy_values(self) -> None:
        """A list containing only falsy values (None, 0, '') should return
        empty string after filtering (line 360)."""
        extractor = StructuredDataExtractor()
        result = extractor._format_value([None, "", 0])
        assert result == ""

    def test_list_with_mixed_truthy_and_falsy(self) -> None:
        """A list with both truthy and falsy values filters out the falsy ones."""
        extractor = StructuredDataExtractor()
        result = extractor._format_value(["hello", None, "world", ""])
        assert result == "hello, world"

    def test_none_returns_empty_string(self) -> None:
        """None input returns empty string."""
        extractor = StructuredDataExtractor()
        result = extractor._format_value(None)
        assert result == ""

    def test_string_value_returned_as_is(self) -> None:
        """A plain string is returned as str()."""
        extractor = StructuredDataExtractor()
        result = extractor._format_value("test value")
        assert result == "test value"

    def test_numeric_value_converted_to_string(self) -> None:
        """A numeric value is converted to string."""
        extractor = StructuredDataExtractor()
        result = extractor._format_value(42)
        assert result == "42"
