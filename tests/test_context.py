# Copyright (C) 2026 Ethereum Foundation
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from pathlib import Path

import pytest

from docc.context import Context, load
from docc.settings import Settings


class TestContext:
    def test_init_empty(self) -> None:
        ctx = Context()
        assert str not in ctx

    def test_init_with_none(self) -> None:
        ctx = Context(None)
        assert str not in ctx

    def test_getitem_missing_raises(self) -> None:
        ctx = Context({})
        with pytest.raises(KeyError):
            ctx[str]

    def test_contains_true(self) -> None:
        ctx = Context({str: "hello"})
        assert str in ctx

    def test_contains_false(self) -> None:
        ctx = Context({})
        assert str not in ctx

    def test_init_validates_types(self) -> None:
        with pytest.raises(ValueError, match="is not an instance"):
            Context({str: 123})

    def test_repr(self) -> None:
        ctx = Context({str: "hello"})
        result = repr(ctx)
        assert "Context" in result
        assert "str" in result

    def test_multiple_types(self) -> None:
        class CustomA:
            pass

        class CustomB:
            pass

        a = CustomA()
        b = CustomB()

        ctx = Context({CustomA: a, CustomB: b})
        assert ctx[CustomA] is a
        assert ctx[CustomB] is b

    def test_subclass_types(self) -> None:
        class Base:
            pass

        class Derived(Base):
            pass

        d = Derived()
        ctx = Context({Derived: d})
        assert ctx[Derived] is d

    def test_base_and_derived_stored_separately(self) -> None:
        class Base:
            pass

        class Derived(Base):
            pass

        b = Base()
        d = Derived()
        ctx = Context({Base: b, Derived: d})
        assert ctx[Base] is b
        assert ctx[Derived] is d

    def test_derived_stored_base_lookup_not_found(self) -> None:
        """
        When a Derived instance is stored under its Derived key,
        looking up by Base raises KeyError because Context uses exact
        type matching on the dict key, not isinstance checks.
        """

        class Base:
            pass

        class Derived(Base):
            pass

        d = Derived()
        ctx = Context({Derived: d})
        assert Base not in ctx
        with pytest.raises(KeyError):
            ctx[Base]


class TestContextLoad:
    def test_load_empty_context_list(self, tmp_path: Path) -> None:
        settings = Settings(
            tmp_path,
            {"tool": {"docc": {"context": []}}},
        )

        result = list(load(settings))
        assert result == []

    def test_load_single_context(self, tmp_path: Path) -> None:
        settings = Settings(
            tmp_path,
            {"tool": {"docc": {"context": ["docc.references.context"]}}},
        )

        result = list(load(settings))
        assert len(result) == 1
        assert result[0][0] == "docc.references.context"

    def test_load_multiple_contexts(self, tmp_path: Path) -> None:
        settings = Settings(
            tmp_path,
            {
                "tool": {
                    "docc": {
                        "context": [
                            "docc.references.context",
                            "docc.search.context",
                        ]
                    }
                }
            },
        )

        result = list(load(settings))
        assert len(result) == 2
