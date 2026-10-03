from web.backend.routers import challenges as router_module
from services import challenge_service


def test_router_resolves_masking_helper_used_by_member_lists_and_curves():
    # A regression guard: mask_email moved to the service module once and the
    # router kept calling it by name, so every members/curves request returned 500.
    assert router_module.mask_email is challenge_service.mask_email
