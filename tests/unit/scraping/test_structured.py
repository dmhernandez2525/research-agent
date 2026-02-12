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
