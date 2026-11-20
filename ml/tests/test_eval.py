from eval.run_eval import matches, summarize


def row(truth, pred, status, domain="plate", tags=()):
    return dict(truth=truth, pred=pred, status=status, domain=domain, tags=list(tags))


def test_matches_treats_truth_unknowns_as_wildcards_only():
    assert matches("MH12AB1234", "MH12AB1234") and not matches("MH12AB1235", "MH12AB1234")
    assert matches("MH12AB1234", "MH12AB?234") and not matches("MH12AB1234", "MH12AB?235")
    assert not matches("MH12AB123", "MH12AB1234")          # length differs


def test_headline_metrics():
    rows = [
        row("A1", "A1", "accept"), row("B2", "B2", "accept"), row("C3", "X3", "accept"),   # 1 accepted-wrong
        row("D4", "D4", "review"), row("E5", "E6", "review"),                              # 1 wasted review
        row("F6", "", "reject", domain="vin", tags=["glare"]),
    ]
    s = summarize(rows)["overall"]
    assert s["n"] == 6 and s["accepted"] == 3 and s["accepted_wrong"] == 1
    assert abs(s["accepted_wrong_rate"] - 1 / 3) < 1e-9 and abs(s["coverage"] - 0.5) < 1e-9
    assert abs(s["review_waste"] - 0.5) < 1e-9 and abs(s["exact"] - 3 / 6) < 1e-9
    by = summarize(rows)
    assert by["by_domain"]["vin"]["reject_rate"] == 1.0
