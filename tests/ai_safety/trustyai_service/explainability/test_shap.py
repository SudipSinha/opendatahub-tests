import math
from http import HTTPStatus
from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.trustyai_service import TrustyAIService

from tests.ai_safety.trustyai_service.explainability.utils import (
    assert_common_explanation_response,
    assert_finite_attributions,
)
from tests.ai_safety.trustyai_service.trustyai_service_utils import TrustyAIServiceClient


@pytest.mark.parametrize(
    "model_namespace, trustyai_service",
    [
        pytest.param(
            {"name": "test-explainability-shap-pvc"},
            {"storage": "pvc"},
            id="test_pvc_storage",
        ),
        pytest.param(
            {"name": "test-explainability-shap-db"},
            {"storage": "db"},
            id="test_db_storage",
        ),
    ],
    indirect=True,
)
@pytest.mark.ai_safety
@pytest.mark.rawdeployment
@pytest.mark.tier1
class TestLocalShap:
    """Verify the public local SHAP explanation contract for regression models."""

    def test_local_shap_preserves_additivity(
        self,
        admin_client: DynamicClient,
        current_client_token: str,
        explainability_trustyai_service: TrustyAIService,
        explainability_gaussian_credit_model: InferenceService,
        explainability_model_url: str,
        explainability_prediction_id: str,
    ) -> None:
        """Given stored target data, when SHAP is requested, then identity-link additivity holds."""
        client = TrustyAIServiceClient(
            token=current_client_token,
            service=explainability_trustyai_service,
            client=admin_client,
        )
        model_name = explainability_gaussian_credit_model.name
        payload: dict[str, Any] = {
            "predictionId": explainability_prediction_id,
            "config": {
                "model": {
                    "base_url": explainability_model_url,
                    "model_name": model_name,
                    "task": "REGRESSION",
                },
                "explainer": {
                    "n_samples": 300,
                    "n_training_rows": 50,
                    "timeout": 120,
                    "seed": 123,
                    "link": "IDENTITY",
                },
            },
        }

        response = client.request_local_explanation(algorithm="shap", payload=payload, timeout=180)

        assert response.status_code == HTTPStatus.OK, response.text
        response_body: dict[str, Any] = response.json()
        assert_common_explanation_response(
            response_body=response_body,
            model_name=model_name,
            prediction_id=explainability_prediction_id,
        )
        assert_finite_attributions(response_body=response_body)

        prediction_output = response_body["prediction_output"]
        base_value = response_body["shap_base_value"]
        linked_prediction_output = response_body["linked_prediction_output"]
        assert all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in (prediction_output, base_value, linked_prediction_output)
        )
        assert all(math.isfinite(float(value)) for value in (prediction_output, base_value, linked_prediction_output))
        attribution_sum = sum(attribution["importance"] for attribution in response_body["attributions"])
        assert math.isclose(
            base_value + attribution_sum,
            linked_prediction_output,
            rel_tol=1e-4,
            abs_tol=1e-3,
        )
        assert math.isclose(prediction_output, linked_prediction_output, rel_tol=1e-4, abs_tol=1e-3)
