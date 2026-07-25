"""Tests for poller helpers, in particular the poll cursor advance logic."""
import poller


def test_advance_ms_increments_millisecond():
    assert poller._advance_ms("2026-07-25T14:35:03.549Z") == "2026-07-25T14:35:03.550Z"


def test_advance_ms_rolls_over_second():
    assert poller._advance_ms("2026-07-25T14:35:03.999Z") == "2026-07-25T14:35:04.000Z"


def test_advance_ms_rolls_over_day():
    assert poller._advance_ms("2026-07-25T23:59:59.999Z") == "2026-07-26T00:00:00.000Z"


def test_advance_ms_result_excludes_source_asset():
    """The whole point: an inclusive createdAfter/takenAfter filter using the
    advanced cursor must not match an asset with the original timestamp."""
    original = "2026-07-25T14:35:03.549Z"
    advanced = poller._advance_ms(original)
    assert advanced > original
