from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEPLOYMENT_WORKFLOWS = (
    REPOSITORY_ROOT / ".github/workflows/deploy.yml",
    REPOSITORY_ROOT / ".github/workflows/build-base.yml",
)


@pytest.mark.parametrize("workflow_path", DEPLOYMENT_WORKFLOWS)
def test_deployment_logs_stay_bounded_and_host_local(workflow_path: Path) -> None:
    workflow = workflow_path.read_text()

    assert "--log-driver=json-file" in workflow
    assert "--log-opt max-size=50m" in workflow
    assert "--log-opt max-file=5" in workflow
    assert "--log-driver=awslogs" not in workflow
    assert "awslogs-group" not in workflow
    assert "LOGS_GROUP" not in workflow
