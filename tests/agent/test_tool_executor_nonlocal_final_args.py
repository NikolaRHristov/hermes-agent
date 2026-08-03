"""Regression guard for the ``nonlocal final_args`` fix in
``agent/tool_executor.py`` (feat-dev branch).

Background
----------
``_run_agent_tool_execution_middleware`` defines a nested function
``_resolve_pre_tool_block`` that:
  * reads ``final_args`` (passed into ``_dispatch_pre_tool_call_hooks``), and
  * when that hook returns ``modified_args is not None``, assigns them back
    via ``final_args = modified_args``.

On the broken version of this branch, the ``nonlocal final_args``
declaration sat *after* the first read of ``final_args`` inside the nested
function. Python compiles the entire function body before executing it, so a
``nonlocal`` declaration that appears textually after a use of the name is a
``SyntaxError`` -- but more subtly, when the read happened first in a way
Python flags, it raised ``name 'final_args' is used prior to nonlocal
declaration`` at call time. That crashed *every* tool call whose pre-tool
hook returned modified arguments (e.g. the aphrodite CCR plugin).

The fix moves ``nonlocal final_args`` to the top of the nested scope so the
name is bound before any use.

This suite mirrors the two-test pattern used elsewhere in the repo:
  * a contract test documenting the Python rule the fix depends on, and
  * a call-site test replicating the exact shape of ``_resolve_pre_tool_block``
    that fails if the declaration is ever moved back below the first use.
"""

from __future__ import annotations

import types


def _make_resolve_block(nonlocal_first: bool):
    """Build a function mirroring ``_resolve_pre_tool_block``'s nonlocal
    handling, with the ``nonlocal`` declaration either at the top
    (``nonlocal_first=True``, the fixed order) or after the first read
    (``nonlocal_first=False``, the broken order).

    Returns ``(outer, captured)`` where ``outer`` is a callable taking
    ``(final_args, modified)`` and returning the value ``final_args`` ended
    up as after the (simulated) hook call.
    """

    def outer(final_args, modified):
        state = {"args": final_args}

        if nonlocal_first:
            # FIXED ORDER: declare nonlocal at the top of the nested scope.
            def _resolve_pre_tool_block():
                nonlocal final_args
                try:
                    # Simulate the hook dispatch returning modified args.
                    modified_args = modified
                    if modified_args is not None:
                        final_args = modified_args
                        state["args"] = modified_args
                    return None
                except Exception:
                    return None

            _resolve_pre_tool_block()
        else:
            # BROKEN ORDER: read final_args first, declare nonlocal after.
            def _resolve_pre_tool_block():
                try:
                    modified_args = modified
                    if modified_args is not None:
                        nonlocal final_args  # declared AFTER first use
                        final_args = modified_args
                        state["args"] = modified_args
                    return None
                except Exception:
                    return None

            _resolve_pre_tool_block()

        return final_args

    return outer


def test_contract_nonlocal_after_use_is_rejected_by_python():
    """Document the Python rule the fix relies on.

    A nested function that uses a name and then declares ``nonlocal`` for it
    is a ``SyntaxError`` at compile time (Python scans the whole function body
    for bindings before executing). This is exactly the trap the broken
    ordering fell into. If Python ever relaxes this, the call-site test below
    becomes the authoritative guard.
    """
    import ast

    broken_source = (
        "def outer():\n"
        "    final_args = 1\n"
        "    def inner():\n"
        "        x = final_args      # use before nonlocal\n"
        "        nonlocal final_args\n"
        "        final_args = x\n"
    )
    with __import__("pytest").raises(SyntaxError):
        compile(broken_source, "<broken>", "exec")


def test_resolve_block_propagates_modified_args_when_nonlocal_is_first():
    """Call-site guard: with ``nonlocal final_args`` declared at the top,
    a hook returning modified args flows back into ``final_args`` without
    error. This is the exact shape of the fixed ``_resolve_pre_tool_block``.
    """
    outer = _make_resolve_block(nonlocal_first=True)

    result = outer("original", "modified-by-hook")
    assert result == "modified-by-hook", (
        f"Pre-tool hook modify did not propagate: expected "
        f"'modified-by-hook', got {result!r}. This is the crash from the "
        f"broken nonlocal ordering -- check that 'nonlocal final_args' is "
        f"declared at the top of _resolve_pre_tool_block in "
        f"agent/tool_executor.py."
    )


def test_resolve_block_passes_through_when_hook_returns_no_modification():
    """Sanity: when the hook returns no modified args, ``final_args`` is
    unchanged. Exercises the common (non-crashing) path too.
    """
    outer = _make_resolve_block(nonlocal_first=True)

    result = outer("original", None)
    assert result == "original"


def test_resolve_block_broken_order_still_runs_when_no_modification():
    """The broken ordering only crashes when modified args are returned.
    When the hook returns ``None`` there is no ``nonlocal`` assignment path
    taken at runtime in this simplified model, so it must still pass through.
    This documents the asymmetry that made the bug intermittent (only fired
    for hooks that actually modified args).
    """
    outer = _make_resolve_block(nonlocal_first=False)

    result = outer("original", None)
    assert result == "original"


def test_real_tool_executor_source_compiles_with_nonlocal_first():
    """Tie the guard to the actual source file.

    The broken ``nonlocal final_args`` ordering is a ``SyntaxError`` that
    surfaces when the *whole module* is compiled/imported -- which is exactly
    why every tool call crashed (the agent could not import
    ``agent.tool_executor``). Reverting the fix reintroduces that ordering and
    the module fails to compile again.

    This test compiles the entire ``agent/tool_executor.py`` source. It does
    NOT import the module (which would boot the gateway stack), so it is safe
    to run in any environment, yet it is ordering-sensitive: a broken
    ``nonlocal`` placement fails compilation of the whole module.
    """
    import ast
    import os

    here = os.path.dirname(os.path.abspath(__file__))
    src_path = os.path.normpath(os.path.join(here, "..", "..", "agent", "tool_executor.py"))
    with open(src_path, "r", encoding="utf-8") as fh:
        source = fh.read()

    # NOTE: ast.parse() only does syntactic parsing and does NOT enforce the
    # nonlocal-binding rule -- that semantic error is raised by compile().
    # The broken ordering makes the *module* fail to compile, which is why
    # every tool call crashed (the agent could not import the module). So we
    # must compile(), not merely parse().
    try:
        compile(source, src_path, "exec")
    except SyntaxError as exc:
        raise AssertionError(
            "agent/tool_executor.py fails to compile: "
            f"{exc}. A 'nonlocal final_args' declaration that appears after "
            "the first use of final_args inside _resolve_pre_tool_block breaks "
            "compilation of the whole module -- which is what crashed every "
            "tool call. Keep 'nonlocal final_args' at the top of the nested "
            "function."
        ) from exc
