import math
from collections.abc import Generator
from contextlib import contextmanager
from http import HTTPStatus
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from kubernetes.dynamic import DynamicClient
from ocp_resources.deployment import Deployment
from ocp_resources.inference_service import InferenceService
from ocp_resources.resource import ResourceEditor
from ocp_resources.trustyai_service import TrustyAIService
from timeout_sampler import TimeoutSampler

from tests.ai_safety.trustyai_service.trustyai_service_utils import TrustyAIServiceClient
from utilities.constants import TRUSTYAI_SERVICE_NAME
from utilities.infra import get_services_by_isvc_label


def assert_common_explanation_response(
    response_body: dict[str, Any],
    model_name: str,
    prediction_id: str,
) -> None:
    """Assert fields common to successful local explanation responses.

    Args:
        response_body (dict[str, Any]): Parsed explanation response.
        model_name (str): Expected public model identity.
        prediction_id (str): Expected stored target identifier.
    """
    assert response_body["prediction_id"] == prediction_id
    assert response_body["model"] == model_name
    assert response_body["task"] == "REGRESSION"


def assert_finite_attributions(response_body: dict[str, Any]) -> None:
    """Assert that an explanation contains finite, uniquely named attributions.

    Args:
        response_body (dict[str, Any]): Parsed explanation response.
    """
    attributions = response_body["attributions"]
    assert isinstance(attributions, list) and attributions

    feature_names: list[str] = []
    for attribution in attributions:
        feature_name = attribution["feature_name"]
        importance = attribution["importance"]
        assert isinstance(feature_name, str) and feature_name
        assert isinstance(importance, (int, float)) and not isinstance(importance, bool)
        assert math.isfinite(float(importance))
        feature_names.append(feature_name)

    assert len(feature_names) == len(set(feature_names))


def get_internal_model_url(
    client: DynamicClient,
    inference_service: InferenceService,
    runtime_name: str,
) -> str:
    """Build a cluster-local URL for the model's HTTP service port.

    Args:
        client (DynamicClient): Kubernetes API client.
        inference_service (InferenceService): Raw-deployment model service.
        runtime_name (str): ServingRuntime name used by the model.

    Returns:
        str: Cluster-local KServe server root URL.

    Raises:
        AssertionError: If no HTTP service port is available.
    """
    services = get_services_by_isvc_label(
        client=client,
        isvc=inference_service,
        runtime_name=runtime_name,
    )
    for service in services:
        ports = service.instance.to_dict().get("spec", {}).get("ports", [])
        candidate_ports = [
            port
            for port in ports
            if (port.get("protocol") or "TCP").upper() == "TCP" and "metrics" not in (port.get("name") or "").lower()
        ]
        http_ports = [
            port
            for port in candidate_ports
            if (port.get("name") or "").lower() in {"http", "http1", "http-port", "rest", "rest-port"}
        ]
        for port in http_ports or candidate_ports:
            service_port = port.get("port")
            if isinstance(service_port, int):
                return f"http://{service.name}.{inference_service.namespace}.svc.cluster.local:{service_port}"

    raise AssertionError(f"No HTTP service port found for InferenceService {inference_service.name}")


@contextmanager
def configure_local_explainers(
    client: DynamicClient,
    trustyai_service: TrustyAIService,
    model_url: str,
) -> Generator[Deployment]:
    """Enable local explainers and allow the cluster-local model host.

    The current TrustyAIService resource wrapper does not expose application
    environment configuration, so this test-only adapter patches the generated
    Deployment and restores it when the context exits.

    Args:
        client (DynamicClient): Kubernetes API client.
        trustyai_service (TrustyAIService): TrustyAI service resource.
        model_url (str): Cluster-local KServe server root URL.

    Yields:
        Deployment: Patched TrustyAI deployment.

    Raises:
        AssertionError: If the expected TrustyAI application container is absent.
    """
    model_authority = urlsplit(url=model_url).netloc
    if not model_authority:
        raise AssertionError(f"Model URL has no host authority: {model_url}")

    deployment = Deployment(
        client=client,
        name=TRUSTYAI_SERVICE_NAME,
        namespace=trustyai_service.namespace,
        ensure_exists=True,
    )
    deployment_body = deployment.instance.to_dict()
    containers = deployment_body["spec"]["template"]["spec"]["containers"]
    service_container = next(
        (container for container in containers if container.get("name") == TRUSTYAI_SERVICE_NAME),
        None,
    )
    if service_container is None:
        raise AssertionError(f"Container {TRUSTYAI_SERVICE_NAME} not found in TrustyAI deployment")

    environment = {entry["name"]: entry for entry in service_container.get("env", []) if entry.get("name")}
    for name, value in {
        "TRUSTYAI_ENABLE_EXPLAINER": "true",
        "TRUSTYAI_ENABLE_EXPLAINER_LOCAL": "true",
        "TRUSTYAI_EXPLAINER_ALLOWED_HOSTS": model_authority,
        "TRUSTYAI_EXPLAINER_ALLOWED_PRIVATE_HOSTS": model_authority,
    }.items():
        environment[name] = {"name": name, "value": value}
    service_container["env"] = list(environment.values())

    patch = {"spec": {"template": {"spec": {"containers": containers}}}}
    with ResourceEditor(patches={deployment: patch}):
        deployment.wait_for_replicas()
        deployment.wait_for_condition(condition="Available", status="True")
        yield deployment

    deployment.wait_for_replicas()


def upload_training_data_and_get_prediction_id(
    client: TrustyAIServiceClient,
    data_path: Path,
    model_name: str,
) -> str:
    """Upload organic background data and return one newly stored target ID.

    Args:
        client (TrustyAIServiceClient): Authenticated TrustyAI client.
        data_path (Path): Existing TrustyAI upload payload.
        model_name (str): Stored model name.

    Returns:
        str: A prediction ID created by this upload.

    Raises:
        AssertionError: If upload fails or no new ID appears before timeout.
    """
    before_response = client.get_inference_ids(model_name=model_name, timeout=30)
    if before_response.status_code in {HTTPStatus.BAD_REQUEST, HTTPStatus.NOT_FOUND}:
        before_ids: set[str] = set()
    else:
        assert before_response.status_code == HTTPStatus.OK, before_response.text
        before_ids = {item["id"] for item in before_response.json().get("ids", [])}

    upload_response = client.upload_data(data_path=str(data_path), timeout=300)
    assert upload_response.status_code == HTTPStatus.OK, upload_response.text

    def new_ids() -> list[str]:
        response = client.get_inference_ids(model_name=model_name, timeout=30)
        if response.status_code in {HTTPStatus.BAD_REQUEST, HTTPStatus.NOT_FOUND}:
            return []
        assert response.status_code == HTTPStatus.OK, response.text
        current_ids = {item["id"] for item in response.json().get("ids", [])}
        return sorted(current_ids - before_ids)

    for ids in TimeoutSampler(wait_timeout=300, sleep=1, func=new_ids):
        if ids:
            return ids[0]

    raise AssertionError(f"No new organic prediction ID appeared for model {model_name}")
