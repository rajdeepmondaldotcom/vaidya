"""Integration tests for the Vaidya FastAPI endpoints.

Tests the HTTP API layer using httpx.AsyncClient with ASGITransport.
NO external dependencies: Redis, Sarvam API, and ChromaDB are all mocked.

Focuses on stateless endpoints (health, schemes) that can work without
infrastructure. Conversation/simulate endpoints require full lifespan
wiring and are covered by test_conversation_flow.py at the orchestrator level.
"""

from __future__ import annotations

import time

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from vaidya import __version__
from vaidya.api.routes.health import router as health_router
from vaidya.api.routes.schemes import router as schemes_router
from vaidya.dependencies import get_schemes as get_schemes_dep
from vaidya.models.scheme import SchemeRecord
from vaidya.schemes.registry import get_schemes

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _build_test_app(schemes: list[SchemeRecord]) -> FastAPI:
    """Build a minimal FastAPI app with health and schemes routers.

    Injects schemes via dependency override so the endpoints work without
    running the full lifespan (which requires Redis, Sarvam API, ChromaDB).
    """
    app = FastAPI(title="Vaidya Test", version=__version__)
    app.include_router(health_router, tags=["health"])
    app.include_router(schemes_router, prefix="/schemes", tags=["schemes"])

    # Override the dependency provider to bypass Request.app.state
    app.dependency_overrides[get_schemes_dep] = lambda: schemes

    return app


@pytest.fixture(scope="module")
def real_schemes() -> list[SchemeRecord]:
    """Load the actual scheme records from the JSON data files."""
    return get_schemes()


@pytest.fixture()
async def client(real_schemes: list[SchemeRecord]):
    """Create an async test client with mocked dependencies."""
    app = _build_test_app(schemes=real_schemes)

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as c:
        yield c


# ===========================================================================
# Health endpoint tests
# ===========================================================================


class TestHealthEndpoint:
    """GET /health -- liveness check."""

    async def test_health_returns_200(self, client: AsyncClient):
        """GET /health returns 200 with status and version."""
        response = await client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "ok"
        assert data["version"] == __version__

    async def test_health_version_matches_package(self, client: AsyncClient):
        """Version in health response matches the package __version__."""
        response = await client.get("/health")
        data = response.json()
        assert data["version"] == "0.1.0"


class TestReadyEndpoint:
    """GET /ready -- readiness check."""

    async def test_ready_returns_200(self, client: AsyncClient):
        """GET /ready returns 200 with ready status."""
        response = await client.get("/ready")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "ready"
        assert "version" in data


# ===========================================================================
# Schemes endpoint tests
# ===========================================================================


class TestListSchemes:
    """GET /schemes -- full scheme catalog."""

    async def test_list_schemes_returns_8(self, client: AsyncClient):
        """GET /schemes returns all 8 schemes from the data directory."""
        response = await client.get("/schemes")
        assert response.status_code == 200

        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 8

    async def test_list_schemes_has_required_fields(self, client: AsyncClient):
        """Each scheme in the list has the required SchemeResponse fields."""
        response = await client.get("/schemes")
        data = response.json()

        required_fields = {
            "scheme_id",
            "canonical_name",
            "coverage_amount_inr",
            "jurisdiction",
            "state_code",
            "description",
        }
        for scheme in data:
            assert required_fields.issubset(scheme.keys()), (
                f"Scheme {scheme.get('scheme_id', '?')} missing fields: "
                f"{required_fields - scheme.keys()}"
            )

    async def test_list_schemes_includes_pmjay(self, client: AsyncClient):
        """PM-JAY (the flagship central scheme) is in the catalog."""
        response = await client.get("/schemes")
        data = response.json()

        scheme_ids = [s["scheme_id"] for s in data]
        assert "PMJAY-2024-v3" in scheme_ids

    async def test_list_schemes_includes_state_schemes(self, client: AsyncClient):
        """Catalog contains a mix of central and state schemes."""
        response = await client.get("/schemes")
        data = response.json()

        jurisdictions = {s["jurisdiction"] for s in data}
        assert "central" in jurisdictions
        assert "state" in jurisdictions


class TestGetSchemeById:
    """GET /schemes/{scheme_id} -- single scheme detail."""

    async def test_get_pmjay_by_id(self, client: AsyncClient):
        """GET /schemes/PMJAY-2024-v3 returns PM-JAY details."""
        response = await client.get("/schemes/PMJAY-2024-v3")
        assert response.status_code == 200

        data = response.json()
        assert data["scheme_id"] == "PMJAY-2024-v3"
        assert "Ayushman" in data["canonical_name"] or "PMJAY" in data["canonical_name"]
        assert data["coverage_amount_inr"] == 500000
        assert data["jurisdiction"] == "central"
        assert data["state_code"] is None

    async def test_get_scheme_returns_description(self, client: AsyncClient):
        """Scheme detail includes a non-empty description."""
        response = await client.get("/schemes/PMJAY-2024-v3")
        data = response.json()
        assert data["description"]
        assert len(data["description"]) > 50  # Should be a real description

    async def test_scheme_not_found_returns_404(self, client: AsyncClient):
        """GET /schemes/INVALID returns 404 with detail message."""
        response = await client.get("/schemes/INVALID-SCHEME-ID")
        assert response.status_code == 404

        data = response.json()
        assert "detail" in data
        assert "not found" in data["detail"].lower()

    async def test_each_known_scheme_is_retrievable(self, client: AsyncClient):
        """Every scheme in the list endpoint is individually retrievable."""
        list_response = await client.get("/schemes")
        all_schemes = list_response.json()

        for scheme in all_schemes:
            detail_response = await client.get(f"/schemes/{scheme['scheme_id']}")
            assert detail_response.status_code == 200, (
                f"Failed to retrieve scheme {scheme['scheme_id']}"
            )
            detail = detail_response.json()
            assert detail["scheme_id"] == scheme["scheme_id"]


# ===========================================================================
# Scheme data integrity tests
# ===========================================================================


class TestSchemeDataIntegrity:
    """Verify the scheme data loaded from JSON is self-consistent."""

    async def test_all_schemes_have_unique_ids(self, client: AsyncClient):
        """No duplicate scheme_ids in the catalog."""
        response = await client.get("/schemes")
        data = response.json()

        ids = [s["scheme_id"] for s in data]
        assert len(ids) == len(set(ids)), f"Duplicate IDs found: {ids}"

    async def test_all_schemes_have_non_negative_coverage(self, client: AsyncClient):
        """Every scheme has coverage_amount_inr >= 0.

        Note: ESIC is contribution-based and has coverage_amount_inr=0,
        which is valid (it is not a fixed coverage scheme).
        """
        response = await client.get("/schemes")
        data = response.json()

        for scheme in data:
            assert scheme["coverage_amount_inr"] >= 0, (
                f"Scheme {scheme['scheme_id']} has negative coverage"
            )

    async def test_central_schemes_have_no_state_code(self, client: AsyncClient):
        """Central schemes have state_code as null."""
        response = await client.get("/schemes")
        data = response.json()

        for scheme in data:
            if scheme["jurisdiction"] == "central":
                assert scheme["state_code"] is None, (
                    f"Central scheme {scheme['scheme_id']} has state_code={scheme['state_code']}"
                )

    async def test_state_schemes_have_state_code(self, client: AsyncClient):
        """State schemes have a non-null state_code."""
        response = await client.get("/schemes")
        data = response.json()

        for scheme in data:
            if scheme["jurisdiction"] == "state":
                assert scheme["state_code"] is not None, (
                    f"State scheme {scheme['scheme_id']} is missing state_code"
                )

    async def test_known_state_scheme_ids(self, client: AsyncClient):
        """Verify the expected state schemes are present."""
        response = await client.get("/schemes")
        data = response.json()

        state_schemes = {
            s["scheme_id"]: s["state_code"] for s in data if s["jurisdiction"] == "state"
        }
        # Expected: Chiranjeevi (RJ), Swasthya Sathi (WB), MJPJAY (MH), Arogya (KA)
        assert "CHIR-RJ-2024-v2" in state_schemes
        assert state_schemes["CHIR-RJ-2024-v2"] == "RJ"

    async def test_known_central_scheme_ids(self, client: AsyncClient):
        """Verify the expected central schemes are present."""
        response = await client.get("/schemes")
        data = response.json()

        central_ids = [s["scheme_id"] for s in data if s["jurisdiction"] == "central"]
        assert "PMJAY-2024-v3" in central_ids
        assert "ESIC-2024-v2" in central_ids
        assert "PMSBY-2024-v2" in central_ids


# ===========================================================================
# Response format tests
# ===========================================================================


class TestSchemeResponseFormat:
    """Verify the API response format matches the SchemeResponse model."""

    async def test_detail_response_has_exact_fields(self, client: AsyncClient):
        """Detail response contains exactly the SchemeResponse fields."""
        response = await client.get("/schemes/PMJAY-2024-v3")
        data = response.json()

        expected_fields = {
            "scheme_id",
            "canonical_name",
            "coverage_amount_inr",
            "jurisdiction",
            "state_code",
            "description",
        }
        assert set(data.keys()) == expected_fields

    async def test_list_items_match_detail_format(self, client: AsyncClient):
        """Each item in the list has the same fields as the detail response."""
        response = await client.get("/schemes")
        data = response.json()

        expected_fields = {
            "scheme_id",
            "canonical_name",
            "coverage_amount_inr",
            "jurisdiction",
            "state_code",
            "description",
        }
        for item in data:
            assert set(item.keys()) == expected_fields


# ===========================================================================
# Edge cases
# ===========================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    async def test_schemes_endpoint_is_idempotent(self, client: AsyncClient):
        """Multiple calls to /schemes return identical data."""
        r1 = await client.get("/schemes")
        r2 = await client.get("/schemes")

        assert r1.json() == r2.json()

    async def test_health_responds_quickly(self, client: AsyncClient):
        """Health endpoint responds in under 100ms (no IO)."""
        start = time.perf_counter()
        response = await client.get("/health")
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert response.status_code == 200
        assert elapsed_ms < 100

    async def test_nonexistent_endpoint_returns_404(self, client: AsyncClient):
        """Request to a non-existent path returns 404."""
        response = await client.get("/nonexistent")
        assert response.status_code == 404

    async def test_method_not_allowed(self, client: AsyncClient):
        """POST to a GET-only endpoint returns 405."""
        response = await client.post("/health")
        assert response.status_code == 405
