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
            {"name": "test-explainability-lime-pvc"},
            {"storage": "pvc"},
            id="test_pvc_storage",
        ),
        pytest.param(
            {"name": "test-explainability-lime-db"},
            {"storage": "db"},
            id="test_db_storage",
        ),
    ],
    indirect=True,
)
@pytest.mark.ai_safety
@pytest.mark.rawdeployment
@pytest.mark.tier1
class TestLocalLime:
    """Verify the public local LIME explanation contract for regression models."""

    def test_local_lime_returns_regression_attributions(
        self,
        admin_client: DynamicClient,
        current_client_token: str,
        explainability_trustyai_service: TrustyAIService,
        explainability_gaussian_credit_model: InferenceService,
        explainability_model_url: str,
        explainability_prediction_id: str,
    ) -> None:
        """Given stored target data, when LIME is requested, then finite attributions are returned."""
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
                    "num_samples": 100,
                    "num_features": 4,
                    "n_training_rows": 50,
                    "timeout": 120,
                    "seed": 123,
                },
            },
        }

        response = client.request_local_explanation(algorithm="lime", payload=payload, timeout=180)

        assert response.status_code == HTTPStatus.OK, response.text
        response_body: dict[str, Any] = response.json()
        assert_common_explanation_response(
            response_body=response_body,
            model_name=model_name,
            prediction_id=explainability_prediction_id,
        )
        assert_finite_attributions(response_body=response_body)
        for field_name in ("score", "local_prediction", "intercept", "prediction_output"):
            field_value = response_body[field_name]
            assert isinstance(field_value, (int, float)) and not isinstance(field_value, bool)
            assert math.isfinite(float(field_value))
