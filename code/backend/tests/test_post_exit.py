"""Where developers went after leaving: contributions elsewhere split by owner."""

from app.stages.context import destinations, elsewhere


def item(repo, n):
    return {"repository": {"nameWithOwner": repo}, "contributions": {"totalCount": n}}


def test_destinations_split_same_owner_own_and_other_repositories():
    c = {
        "commitContributionsByRepository": [item("Org/project", 50), item("org/other-tool", 4), item("Alice/dotfiles", 7)],
        "pullRequestContributionsByRepository": [item("someone/lib", 2), item("org/other-tool", 1)],
    }
    assert destinations(c, "org/Project", "alice") == (5, 7, 2)
    assert elsewhere(c, "org/Project") == (14, 3)  # the split adds up to the total
