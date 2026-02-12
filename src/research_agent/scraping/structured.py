"""Structured data extraction from HTML (JSON-LD and Schema.org).

Extracts machine-readable structured data embedded in web pages,
including JSON-LD script blocks and Schema.org metadata, and formats
it for use alongside extracted text content.
"""

from __future__ import annotations

import json
import re
from typing import Any, ClassVar

import structlog
from pydantic import BaseModel, Field

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class StructuredDataItem(BaseModel):
    """A single piece of extracted structured data."""

    schema_type: str = Field(
        default="Unknown", description="Schema.org @type (e.g. Article, Product)."
    )
    properties: dict[str, Any] = Field(
        default_factory=dict, description="Extracted key-value properties."
    )
    source: str = Field(
        default="json_ld", description="Extraction source: json_ld or microdata."
    )


class StructuredDataResult(BaseModel):
    """Result of structured data extraction from a page."""

    items: list[StructuredDataItem] = Field(default_factory=list)
    has_structured_data: bool = Field(
        default=False, description="Whether any structured data was found."
    )
    schema_types: list[str] = Field(
        default_factory=list, description="Unique Schema.org types found."
    )
    quality_boost: float = Field(
        default=0.0,
        ge=0.0,
        le=0.2,
        description="Score boost for content with rich structured data.",
    )


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------


class StructuredDataExtractor:
    """Extracts JSON-LD and Schema.org structured data from HTML.

    Parses ``<script type="application/ld+json">`` blocks and extracts
    relevant properties. Supports Article, NewsArticle, BlogPosting,
    Product, Person, Organization, and other common Schema.org types.

    Attributes:
        max_items: Maximum number of structured data items to extract.
    """

    # Properties to extract per Schema.org type
    PROPERTY_MAP: ClassVar[dict[str, list[str]]] = {
        "Article": [
            "headline",
            "description",
            "author",
            "datePublished",
            "dateModified",
            "publisher",
            "wordCount",
            "articleSection",
        ],
        "NewsArticle": [
            "headline",
            "description",
            "author",
            "datePublished",
            "dateModified",
            "publisher",
            "dateline",
        ],
        "BlogPosting": [
            "headline",
            "description",
            "author",
            "datePublished",
            "dateModified",
            "wordCount",
        ],
        "Product": [
            "name",
            "description",
            "brand",
            "offers",
            "aggregateRating",
            "sku",
            "category",
        ],
        "Person": [
            "name",
            "jobTitle",
            "affiliation",
            "url",
            "sameAs",
        ],
        "Organization": [
            "name",
            "description",
            "url",
            "logo",
            "sameAs",
        ],
        "Review": [
            "reviewBody",
            "author",
            "reviewRating",
            "itemReviewed",
            "datePublished",
        ],
        "HowTo": [
            "name",
            "description",
            "step",
            "totalTime",
            "estimatedCost",
        ],
        "FAQPage": [
            "mainEntity",
            "name",
            "description",
        ],
        "WebPage": [
            "name",
            "description",
            "url",
            "datePublished",
            "dateModified",
        ],
    }

    # Quality boost per schema type (richer data = higher boost)
    TYPE_BOOST: ClassVar[dict[str, float]] = {
        "Article": 0.10,
        "NewsArticle": 0.10,
        "BlogPosting": 0.08,
        "Product": 0.12,
        "Review": 0.08,
        "FAQPage": 0.15,
        "HowTo": 0.12,
        "Person": 0.05,
        "Organization": 0.05,
        "WebPage": 0.03,
    }

    def __init__(self, max_items: int = 10) -> None:
        """Initialize the structured data extractor.

        Args:
            max_items: Maximum structured data items to extract per page.
        """
        self.max_items = max_items

    def extract(self, html: str) -> StructuredDataResult:
        """Extract structured data from HTML.

        Args:
            html: Raw HTML content.

        Returns:
            A ``StructuredDataResult`` with extracted items and metadata.
        """
        if not html or not html.strip():
            return StructuredDataResult()

        items: list[StructuredDataItem] = []

        # Extract from JSON-LD blocks
        json_ld_items = self._extract_json_ld(html)
        items.extend(json_ld_items)

        # Truncate to max_items
        items = items[: self.max_items]

        # Compute metadata
        schema_types = sorted({item.schema_type for item in items})
        quality_boost = self._calculate_boost(items)
        has_data = len(items) > 0

        if has_data:
            logger.info(
                "structured_data_found",
                item_count=len(items),
                types=schema_types,
                quality_boost=round(quality_boost, 3),
            )

        return StructuredDataResult(
            items=items,
            has_structured_data=has_data,
            schema_types=schema_types,
            quality_boost=round(min(quality_boost, 0.2), 3),
        )

    def format_for_content(self, result: StructuredDataResult) -> str:
        """Format extracted structured data as text to append to content.

        Produces a human-readable summary of structured data that can be
        appended to extracted text for downstream LLM processing.

        Args:
            result: The extraction result.

        Returns:
            Formatted text block, or empty string if no data.
        """
        if not result.has_structured_data:
            return ""

        lines = ["", "[STRUCTURED DATA]"]
        for item in result.items:
            lines.append(f"Type: {item.schema_type}")
            for key, value in item.properties.items():
                formatted = self._format_value(value)
                if formatted:
                    lines.append(f"  {key}: {formatted}")
            lines.append("")

        return "\n".join(lines).rstrip()

    def _extract_json_ld(self, html: str) -> list[StructuredDataItem]:
        """Extract items from JSON-LD script blocks.

        Args:
            html: Raw HTML.

        Returns:
            List of extracted structured data items.
        """
        items: list[StructuredDataItem] = []

        # Find all JSON-LD script blocks
        pattern = (
            r'<script\s+type\s*=\s*["\']application/ld\+json["\'][^>]*>(.*?)</script>'
        )
        blocks = re.findall(pattern, html, re.DOTALL | re.IGNORECASE)

        for block in blocks:
            try:
                data = json.loads(block.strip())
            except (json.JSONDecodeError, ValueError):
                continue

            # Handle @graph arrays
            if isinstance(data, dict) and "@graph" in data:
                graph_items = data["@graph"]
                if isinstance(graph_items, list):
                    for graph_item in graph_items:
                        item = self._parse_json_ld_item(graph_item)
                        if item:
                            items.append(item)
                continue

            # Handle single item or array at top level
            if isinstance(data, list):
                for entry in data:
                    item = self._parse_json_ld_item(entry)
                    if item:
                        items.append(item)
            elif isinstance(data, dict):
                item = self._parse_json_ld_item(data)
                if item:
                    items.append(item)

        return items

    def _parse_json_ld_item(self, data: dict[str, Any]) -> StructuredDataItem | None:
        """Parse a single JSON-LD object into a StructuredDataItem.

        Args:
            data: A parsed JSON-LD object.

        Returns:
            A StructuredDataItem, or None if the object lacks a type.
        """
        if not isinstance(data, dict):
            return None

        schema_type = data.get("@type", "")
        if isinstance(schema_type, list):
            schema_type = schema_type[0] if schema_type else "Unknown"
        if not schema_type:
            return None

        # Get relevant properties for this type
        prop_names = self.PROPERTY_MAP.get(schema_type, [])
        if not prop_names:
            # Use generic extraction for unknown types
            prop_names = ["name", "description", "url", "datePublished"]

        properties: dict[str, Any] = {}
        for prop in prop_names:
            value = data.get(prop)
            if value is not None:
                properties[prop] = self._simplify_value(value)

        if not properties:
            return None

        return StructuredDataItem(
            schema_type=schema_type,
            properties=properties,
            source="json_ld",
        )

    def _simplify_value(self, value: Any) -> Any:
        """Simplify nested JSON-LD values for storage.

        Extracts names from nested objects and flattens lists of objects.

        Args:
            value: A JSON-LD value (string, dict, list, etc.)

        Returns:
            Simplified value suitable for the properties dict.
        """
        if isinstance(value, dict):
            # Extract the most useful field from nested objects
            for key in ("name", "@value", "value", "text"):
                if key in value:
                    return value[key]
            return str(value.get("@type", value))
        if isinstance(value, list):
            return [self._simplify_value(v) for v in value[:5]]
        return value

    def _format_value(self, value: Any) -> str:
        """Format a property value for text output.

        Args:
            value: Property value to format.

        Returns:
            Formatted string, or empty string for None/empty values.
        """
        if value is None:
            return ""
        if isinstance(value, list):
            formatted = [str(v) for v in value if v]
            return ", ".join(formatted) if formatted else ""
        return str(value)

    def _calculate_boost(self, items: list[StructuredDataItem]) -> float:
        """Calculate quality score boost from structured data.

        Args:
            items: Extracted structured data items.

        Returns:
            Quality boost value (0.0 to 0.2).
        """
        if not items:
            return 0.0

        # Use the highest boost among all found types
        max_boost = 0.0
        for item in items:
            boost = self.TYPE_BOOST.get(item.schema_type, 0.02)
            max_boost = max(max_boost, boost)

        # Add a small bonus for multiple items (capped)
        multi_bonus = min(0.05, (len(items) - 1) * 0.01)

        return min(0.2, max_boost + multi_bonus)
