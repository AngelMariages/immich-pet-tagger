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


# ---------------------------------------------------------------------------
# classify_outcome
# ---------------------------------------------------------------------------

def test_classify_outcome_unknown():
    assert poller.classify_outcome("unknown", 0.99, "2026-06-01T00:00:00Z", {}) == "unknown"


def test_classify_outcome_out_of_range_beats_low_confidence():
    """A low-confidence guess for a pet who was not even in range on that date must
    be dropped as out_of_range, not surfaced as a low-confidence review candidate."""
    cfg = {"since": "2025-01-01", "until": "2025-12-31"}
    outcome = poller.classify_outcome("Dobby", 0.67, "2026-06-01T00:00:00Z", cfg, threshold=0.8)
    assert outcome == "out_of_range"


def test_classify_outcome_out_of_range_beats_confident():
    cfg = {"since": "2025-01-01", "until": "2025-12-31"}
    outcome = poller.classify_outcome("Dobby", 0.95, "2026-06-01T00:00:00Z", cfg, threshold=0.8)
    assert outcome == "out_of_range"


def test_classify_outcome_low_confidence_when_in_range():
    cfg = {"since": "2025-01-01", "until": "2027-12-31"}
    outcome = poller.classify_outcome("Dobby", 0.67, "2026-06-01T00:00:00Z", cfg, threshold=0.8)
    assert outcome == "low_confidence"


def test_classify_outcome_confident_when_in_range():
    cfg = {"since": "2025-01-01", "until": "2027-12-31"}
    outcome = poller.classify_outcome("Dobby", 0.95, "2026-06-01T00:00:00Z", cfg, threshold=0.8)
    assert outcome == "confident"


def test_classify_outcome_no_date_bounds_never_out_of_range():
    outcome = poller.classify_outcome("Dobby", 0.67, "2026-06-01T00:00:00Z", {}, threshold=0.8)
    assert outcome == "low_confidence"
