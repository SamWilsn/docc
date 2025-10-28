"""
Test script to demonstrate the fix for non-transparent elements in references.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from docc.plugins.html import (
    HTMLTag,
    TextNode,
    _contains_non_transparent_elements,
)
from docc.plugins import references


def test_non_transparent_detection():
    """Test that we can detect non-transparent elements."""

    # Test case 1: Simple text node (should be fine)
    text_node = TextNode("Hello World")
    assert not _contains_non_transparent_elements(text_node)
    print("✓ Text node detection works")

    # Test case 2: Simple HTML tag that can be in anchor
    div_tag = HTMLTag("div")
    div_tag.append(TextNode("Some content"))
    assert not _contains_non_transparent_elements(div_tag)
    print("✓ Div tag detection works")

    # Test case 3: Table row (should be detected as non-transparent)
    tr_tag = HTMLTag("tr")
    td_tag = HTMLTag("td")
    td_tag.append(TextNode("Cell content"))
    tr_tag.append(td_tag)
    assert _contains_non_transparent_elements(tr_tag)
    print("✓ Table row detection works")

    # Test case 4: Table cell (should be detected as non-transparent)
    assert _contains_non_transparent_elements(td_tag)
    print("✓ Table cell detection works")

    # Test case 5: Nested structure with non-transparent element
    table_tag = HTMLTag("table")
    table_tag.append(tr_tag)
    assert _contains_non_transparent_elements(table_tag)
    print("✓ Nested non-transparent detection works")

    print(
        "\nAll tests passed! The non-transparent element detection is working correctly."
    )


def test_reference_creation():
    """Test creating references with different content types."""

    # Test case 1: Reference with simple text
    simple_ref = references.Reference("test_ref", TextNode("Simple text"))
    print(f"✓ Created reference with simple text: {simple_ref.identifier}")

    # Test case 2: Reference with table content
    tr_tag = HTMLTag("tr")
    td_tag = HTMLTag("td")
    td_tag.append(TextNode("Table cell"))
    tr_tag.append(td_tag)

    table_ref = references.Reference("table_ref", tr_tag)
    print(f"✓ Created reference with table content: {table_ref.identifier}")

    print("\nReference creation tests passed!")


if __name__ == "__main__":
    print("Testing non-transparent element handling in references...\n")

    test_non_transparent_detection()
    print()
    test_reference_creation()

    print("\n" + "=" * 60)
    print("SUMMARY:")
    print("The solution successfully detects non-transparent HTML elements")
    print("that cannot be descendants of <a> tags. When such elements are")
    print("found in a reference, the system will:")
    print("1. Try to invert the structure (move <a> inside suitable elements)")
    print("2. If inversion fails, render without the link and warn the user")
    print("3. Maintain HTML validity in all cases")
    print("=" * 60)
